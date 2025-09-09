import streamlit as st
import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError
from utils import haversine
import optimization as opt
from optimization import optimize
from visualization import plot_network, summary

st.set_page_config(page_title="Warehouse Network Optimizer", layout="wide")

ROAD_FACTOR = opt.ROAD_FACTOR

def build_lane_df(res, scn):
    lanes = []
    for r in res["assigned"].itertuples():
        wlon, wlat = res["centers"][int(r.Warehouse)]
        cost = r.DemandLbs * r.DistMi * scn["rate_out"]
        lanes.append(dict(
            lane_type="outbound",
            brand=getattr(r, "Brand", ""),
            country=getattr(r, "Country", ""),
            origin_lon=wlon, origin_lat=wlat,
            dest_lon=r.Longitude, dest_lat=r.Latitude,
            distance_mi=r.DistMi,
            weight_lbs=r.DemandLbs,
            rate=scn["rate_out"],
            cost=cost,
        ))

    tier1_coords = res.get("tier1_coords") or []
    center_to_t1_idx = res.get("center_to_t1_idx")
    demand_per_wh = res.get("demand_per_wh", [])
    if tier1_coords and center_to_t1_idx is not None:
        for j, ((wlon, wlat), dem) in enumerate(zip(res["centers"], demand_per_wh)):
            t_idx = center_to_t1_idx[j]
            tx, ty = tier1_coords[t_idx]
            dist = haversine(tx, ty, wlon, wlat) * ROAD_FACTOR
            if dist > 1e-6 and dem > 0:
                lanes.append(dict(
                    lane_type="transfer",
                    origin_lon=tx, origin_lat=ty,
                    dest_lon=wlon, dest_lat=wlat,
                    distance_mi=dist,
                    weight_lbs=dem,
                    rate=scn["trans_rate"],
                    cost=dem * dist * scn["trans_rate"],
                ))

    if scn.get("inbound_on") and scn.get("inbound_pts"):
        if tier1_coords and res.get("tier1_downstream_dem"):
            t1_downstream_dem = res["tier1_downstream_dem"]
            for slon, slat, pct in scn["inbound_pts"]:
                for (tx, ty), handled in zip(tier1_coords, t1_downstream_dem):
                    dist = haversine(slon, slat, tx, ty) * ROAD_FACTOR
                    wt = handled * pct
                    if wt > 0:
                        lanes.append(dict(
                            lane_type="inbound",
                            origin_lon=slon, origin_lat=slat,
                            dest_lon=tx, dest_lat=ty,
                            distance_mi=dist,
                            weight_lbs=wt,
                            rate=scn["in_rate"],
                            cost=wt * dist * scn["in_rate"],
                        ))
        else:
            for slon, slat, pct in scn["inbound_pts"]:
                for (wlon, wlat), wh_dem in zip(res["centers"], demand_per_wh):
                    dist = haversine(slon, slat, wlon, wlat) * ROAD_FACTOR
                    wt = wh_dem * pct
                    if wt > 0:
                        lanes.append(dict(
                            lane_type="inbound",
                            origin_lon=slon, origin_lat=slat,
                            dest_lon=wlon, dest_lat=wlat,
                            distance_mi=dist,
                            weight_lbs=wt,
                            rate=scn["in_rate"],
                            cost=wt * dist * scn["in_rate"],
                        ))
    return pd.DataFrame(lanes)

if "scenarios" not in st.session_state:
    st.session_state["scenarios"]={}

def _num_input(scn,key,label,default,fmt="%.4f",**kw):
    scn.setdefault(key,default)
    scn[key]=st.number_input(label,value=scn[key],format=fmt,
                             key=f"{key}_{scn['_name']}",**kw)

def sidebar(scn):
    name=scn["_name"]
    with st.sidebar:
        st.header(f"Inputs — {name}")
        with st.expander("🗂️ Demand & Candidate Files",True):
            up=st.file_uploader("Demand CSV (Longitude, Latitude, DemandLbs, Brand, Country, CurrWH_Lon, CurrWH_Lat) — one row per customer-brand",
                                key=f"dem_{name}")
            if up:
                scn["demand_file"]=up
                # Detect brands for constraints UI
                try:
                    _df = pd.read_csv(up)
                    brands = sorted([str(x) for x in _df.get("Brand", pd.Series([])).dropna().unique().tolist()])
                except Exception:
                    brands = []
                scn["brands"] = brands
            if "demand_file" not in scn:
                st.info("Upload demand file to continue.")
                return False
            if st.checkbox("Preview demand",key=f"pre_{name}"):
                st.dataframe(pd.read_csv(scn["demand_file"]).head())

            cand_up=st.file_uploader("Candidate WH CSV (lon,lat[,cost/sqft])",
                                     key=f"cand_{name}")
            if cand_up is not None:
                if cand_up:
                    scn["cand_file"]=cand_up
                else:
                    scn.pop("cand_file",None)
            scn["restrict_cand"]=st.checkbox("Restrict to candidates",
                                             value=scn.get("restrict_cand",False),
                                             key=f"rc_{name}")

        with st.expander("🇨🇦 Canada Routing Controls", False):
            scn["can_en"] = st.checkbox("Enable Canada routing (Toronto-facing rule)", value=scn.get("can_en", False), key=f"can_en_{name}")
            cols = st.columns(2)
            scn["can_lon"] = cols[0].number_input("Longitude threshold (east faces Canada)", value=float(scn.get("can_lon", -105.0)), format="%.3f", key=f"can_lon_{name}")
            scn["can_wh_lon"] = cols[1].number_input("Canada WH Lon", value=float(scn.get("can_wh_lon", -79.3832)), format="%.6f", key=f"can_wh_lon_{name}")
            scn["can_wh_lat"] = st.number_input("Canada WH Lat", value=float(scn.get("can_wh_lat", 43.6532)), format="%.6f", key=f"can_wh_lat_{name}")
            st.caption("All rows with Country=='CAN' and customer Longitude ≥ threshold are forced to the Canada WH. Other CAN rows (west of threshold) and USA rows use the nearest US center.")

        with st.expander("🏗️ Current-State Calibration", False):
            scn["use_current_state"] = st.checkbox("Calibrate to current state (force flows using CurrWH_Lon/CurrWH_Lat)", value=scn.get("use_current_state", False), key=f"cs_{name}")
            if scn["use_current_state"]:
                st.caption("Centers derived from current-state columns; brand & Canada rules are ignored.")

        with st.expander("🏷️ Brand Fulfillment Constraints (optional)", False):
            st.caption("Enter allowed warehouse coordinates (lon,lat per line) for each brand. Leave blank for no restriction.")
            brands = scn.get("brands", [])
            brand_allowed = {}
            if brands:
                for b in brands:
                    key = f"brand_allowed_{b}_{name}"
                    txt_default = scn.get(key, "")
                    txt = st.text_area(f"{b} — allowed warehouses (lon,lat per line)", value=txt_default, key=key, height=80)
                    scn[key] = txt
                    pairs = []
                    for ln in txt.splitlines():
                        try:
                            lon, lat = map(float, ln.split(","))
                            pairs.append([lon, lat])
                        except Exception:
                            continue
                    brand_allowed[b] = pairs
            scn["brand_allowed_sites"] = brand_allowed

        with st.expander("💰 Cost Parameters",False):
            st.subheader("Transportation $ / lb-mile")
            _num_input(scn,"rate_out","Outbound",0.35)
            _num_input(scn,"in_rate","Inbound",0.30)
            _num_input(scn,"trans_rate","Transfer",0.32)
            st.subheader("Warehouse")
            _num_input(scn,"sqft_per_lb","Sq ft per lb",0.02)
            _num_input(scn,"cost_sqft","$/sq ft / yr",6.0,"%.2f")
            _num_input(scn,"fixed_wh_cost","Fixed $",250000.0,"%.0f",step=50000.0)

        with st.expander("🔢 Warehouse Count",False):
            scn["auto_k"]=st.checkbox("Optimize k",value=scn.get("auto_k",True),
                                      key=f"ak_{name}")
            if scn["auto_k"]:
                scn["k_rng"]=st.slider("k range",1,30,scn.get("k_rng",(3,6)),
                                       key=f"kr_{name}")
            else:
                _num_input(scn,"k_fixed","k",4,"%.0f",min_value=1,max_value=30)

        with st.expander("📍 Locations",False):
            st.subheader("Fixed Warehouses (US)")
            fixed_txt=st.text_area("lon,lat per line",value=scn.get("fixed_txt",""),
                                   key=f"fx_{name}",height=80)
            scn["fixed_txt"]=fixed_txt
            fixed_centers=[]
            for ln in fixed_txt.splitlines():
                try:
                    lon,lat=map(float,ln.split(","))
                    fixed_centers.append([lon,lat])
                except Exception:
                    continue
            scn["fixed_centers"]=fixed_centers

            st.subheader("Inbound supply points")
            scn["inbound_on"]=st.checkbox("Enable inbound",
                                          value=scn.get("inbound_on",False),
                                          key=f"inb_{name}")
            inbound_pts=[]
            if scn["inbound_on"]:
                sup_txt=st.text_area("lon,lat,percent (0-100)",
                                     value=scn.get("sup_txt",""),
                                     key=f"sup_{name}",height=80)
                scn["sup_txt"]=sup_txt
                for ln in sup_txt.splitlines():
                    try:
                        lon,lat,pct=map(float,ln.split(","))
                        inbound_pts.append([lon,lat,pct/100.0])
                    except Exception:
                        continue
            scn["inbound_pts"]=inbound_pts

        with st.expander("🏬 RDC / SDC (up to 3)",False):
            for idx in range(1,4):
                cols=st.columns([1,4])
                en=cols[0].checkbox(f"{idx}",key=f"rdc_en_{name}_{idx}",
                                     value=scn.get(f"rdc{idx}_en",False))
                scn[f"rdc{idx}_en"]=en
                if en:
                    with cols[1]:
                        lon=st.number_input("Longitude",format="%.6f",
                                            value=float(scn.get(f"rdc{idx}_lon",0.0)),
                                            key=f"lon_{name}_{idx}")
                        lat=st.number_input("Latitude",format="%.6f",
                                            value=float(scn.get(f"rdc{idx}_lat",0.0)),
                                            key=f"lat_{name}_{idx}")
                        typ=st.radio("Type",["RDC","SDC"],horizontal=True,
                                     key=f"typ_{name}_{idx}",
                                     index=0 if scn.get(f"rdc{idx}_typ","RDC")=="RDC" else 1)
                        scn[f"rdc{idx}_lon"]=lon
                        scn[f"rdc{idx}_lat"]=lat
                        scn[f"rdc{idx}_typ"]=typ
            _num_input(scn,"rdc_sqft_per_lb","RDC Sq ft per lb",
                       scn.get("rdc_sqft_per_lb",scn.get("sqft_per_lb",0.02)))
            _num_input(scn,"rdc_cost_sqft","RDC $/sq ft / yr",
                       scn.get("rdc_cost_sqft",scn.get("cost_sqft",6.0)),"%.2f")

        with st.expander("📦 Service Level (optional)", False):
            scn["sl_enforce"] = st.checkbox("Enforce minimum service levels", value=scn.get("sl_enforce", False), key=f"sl_enf_{name}")
            cols = st.columns(2)
            scn["sl_0_350"] = cols[0].number_input("0–350 mi (Next day by 7AM) — min %", min_value=0.0, max_value=100.0, step=1.0, value=float(scn.get("sl_0_350", 0.0)), key=f"sl0350_{name}")
            scn["sl_351_500"] = cols[1].number_input("351–500 mi (Next day by 10AM) — min %", min_value=0.0, max_value=100.0, step=1.0, value=float(scn.get("sl_351_500", 0.0)), key=f"sl351500_{name}")
            cols2 = st.columns(2)
            scn["sl_501_700"] = cols2[0].number_input("501–700 mi (Next day EOD) — min %", min_value=0.0, max_value=100.0, step=1.0, value=float(scn.get("sl_501_700", 0.0)), key=f"sl501700_{name}")
            scn["sl_701p"] = cols2[1].number_input("701+ mi (2 day +) — min %", min_value=0.0, max_value=100.0, step=1.0, value=float(scn.get("sl_701p", 0.0)), key=f"sl701p_{name}")
            tot = scn["sl_0_350"] + scn["sl_351_500"] + scn["sl_501_700"] + scn["sl_701p"]
            if tot > 100.0:
                st.warning(f"Sum of minimums is {tot:.1f}% (> 100%). Exact feasibility may be impossible; I'll minimize total shortfall.")

        st.markdown("---")
        if st.button("🚀 Run solver",key=f"run_{name}"):
            st.session_state["run_target"]=name
    return True

tab_names=list(st.session_state["scenarios"].keys())+["➕ New scenario"]
tabs=st.tabs(tab_names)
for i,tab in enumerate(tabs[:-1]):
    name=tab_names[i]
    scn=st.session_state["scenarios"][name]
    scn["_name"]=name
    with tab:
        if not sidebar(scn):
            continue

        k_vals=(list(range(int(scn["k_rng"][0]),int(scn["k_rng"][1])+1))
                if scn.get("auto_k",True) else [int(scn["k_fixed"])])

        if st.session_state.get("run_target")==name:
            with st.spinner("Optimizing…"):
                df=pd.read_csv(scn["demand_file"]).dropna(subset=["Longitude","Latitude","DemandLbs"])

                candidate_sites=candidate_costs=None
                if scn.get("cand_file"):
                    try:
                        cf=pd.read_csv(scn["cand_file"],header=None).dropna(subset=[0,1])
                        if scn.get("restrict_cand"):
                            candidate_sites=cf.iloc[:,:2].values.tolist()
                        if cf.shape[1]>=3:
                            candidate_costs={(round(r[0],6),round(r[1],6)):r[2]
                                             for r in cf.itertuples(index=False)}
                    except EmptyDataError:
                        scn.pop("cand_file",None)

                # build rdc_list
                rdc_list=[{"coords":[scn[f"rdc{i}_lon"],scn[f"rdc{i}_lat"]],
                           "is_sdc":scn.get(f"rdc{i}_typ","RDC")=="SDC"}
                          for i in range(1,4) if scn.get(f"rdc{i}_en")]

                sl_targets = {
                    "by7": float(scn.get("sl_0_350", 0.0))/100.0,
                    "by10": float(scn.get("sl_351_500", 0.0))/100.0,
                    "eod": float(scn.get("sl_501_700", 0.0))/100.0,
                    "2day": float(scn.get("sl_701p", 0.0))/100.0,
                }

                res=optimize(
                    df=df,
                    k_vals=k_vals,
                    rate_out=scn["rate_out"],
                    sqft_per_lb=scn["sqft_per_lb"],
                    cost_sqft=scn["cost_sqft"],
                    fixed_cost=scn["fixed_wh_cost"],
                    consider_inbound=scn["inbound_on"],
                    inbound_rate_mile=scn["in_rate"],
                    inbound_pts=scn["inbound_pts"],
                    fixed_centers=scn["fixed_centers"],
                    rdc_list=rdc_list,
                    transfer_rate_mile=scn["trans_rate"],
                    rdc_sqft_per_lb=scn["rdc_sqft_per_lb"],
                    rdc_cost_per_sqft=scn["rdc_cost_sqft"],
                    candidate_sites=candidate_sites,
                    restrict_cand=scn.get("restrict_cand",False),
                    candidate_costs=candidate_costs,
                    service_level_targets=sl_targets,
                    enforce_service_levels=bool(scn.get("sl_enforce", False)),
                    # Prior features:
                    current_state=bool(scn.get("use_current_state", False)),
                    brand_col="Brand",
                    curr_wh_lon_col="CurrWH_Lon",
                    curr_wh_lat_col="CurrWH_Lat",
                    brand_allowed_sites=scn.get("brand_allowed_sites", {}),
                    # Canada routing controls
                    country_col="Country",
                    canada_enabled=bool(scn.get("can_en", False)),
                    canada_threshold_lon=float(scn.get("can_lon", -105.0)),
                    canada_wh=[float(scn.get("can_wh_lon", -79.3832)), float(scn.get("can_wh_lat", 43.6532))]
                )

            plot_network(res["assigned"],res["centers"])
            summary(res["assigned"],res["total_cost"],res["out_cost"],
                    res["in_cost"],res["trans_cost"],res["wh_cost"],
                    res["centers"],res["demand_per_wh"],
                    scn["sqft_per_lb"],bool(res.get("rdc_list")),
                    scn["inbound_on"],res["trans_cost"]>0)

            st.subheader("Service Levels Achieved")
            sl = res.get("service_levels", {})
            tgt = res.get("sl_targets", {})
            def _pct(x): return f"{(100.0*float(x)):.1f}%"

            a_cols = st.columns(4)
            a_cols[0].metric("≤ 350 mi  (Next day 7AM)", _pct(sl.get("by7", 0.0)), (_pct(sl.get("by7", 0.0)) if not tgt else f"Target {_pct(tgt.get('by7', 0.0))}"))
            a_cols[1].metric("351–500 mi (Next day 10AM)", _pct(sl.get("by10", 0.0)), (_pct(sl.get("by10", 0.0)) if not tgt else f"Target {_pct(tgt.get('by10', 0.0))}"))
            a_cols[2].metric("501–700 mi (Next day EOD)", _pct(sl.get("eod", 0.0)), (_pct(sl.get("eod", 0.0)) if not tgt else f"Target {_pct(tgt.get('eod', 0.0))}"))
            a_cols[3].metric("701+ mi    (2 day +)", _pct(sl.get("2day", 0.0)), (_pct(sl.get("2day", 0.0)) if not tgt else f"Target {_pct(tgt.get('2day', 0.0))}"))

            if float(res.get("sl_penalty", 0.0)) > 0.0 and scn.get("sl_enforce", False):
                st.info("A service-level penalty was applied to prefer solutions closer to your minimums. If exact targets are infeasible, the solver selects the least-shortfall option.")

            lanes_df=build_lane_df(res,scn)
            st.download_button("📥 Download lane-level calculations (CSV)",
                               lanes_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"{name}_lanes.csv",mime="text/csv")

with tabs[-1]:
    new_name=st.text_input("Scenario name")
    if st.button("Create scenario"):
        if new_name and new_name not in st.session_state["scenarios"]:
            st.session_state["scenarios"][new_name]={}
            if hasattr(st,"rerun"):
                st.rerun()
            else:
                st.experimental_rerun()