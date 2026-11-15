"""Events & Alerts view (Phase 5F): events, alerts, notifications, Qwen.

Thin Streamlit wrapper over the events store. Reads pinned analytical
outputs and the events database; computes no analytics. All lists are
bounded and paged; every write (reviews, ack/resolve, subscriptions)
goes through the audited store helpers.
"""

from __future__ import annotations

from pathlib import Path

from .. import settings
from ..agents.granite import GraniteProvider
from ..agents.providers import ProviderChain, QwenProvider
from ..events import model, notify, qwen, store
from . import event_report, frames, query

KEY_EV_STATE = "cal_ev_state"
KEY_EV_SEVERITY = "cal_ev_severity"
KEY_EV_PAGE = "cal_ev_page"
KEY_EV_SELECTED = "cal_ev_selected"
KEY_AL_STATUS = "cal_al_status"
KEY_AL_TYPE = "cal_al_type"
KEY_AL_PAGE = "cal_al_page"
KEY_AL_SELECTED = "cal_al_selected"
KEY_EV_SECTION = "cal_ev_section"

PAGE_SIZE = 50

METHODOLOGY_NOTE = ("Events group satellite observations over time; they are "
                    "not confirmed wildfires, perimeters, or incidents. "
                    "Alerts classify the activity signal, not fire danger. "
                    "Resolved never means contained or extinguished.")


def _db_ok() -> bool:
    return query.events_db_status()["available"]


def _connect():
    db_file = str(settings.resolve_repo_path(
        settings.get_setting("CAL_EVENTS_DB_PATH", settings.CAL_EVENTS_DB_PATH)))
    conn = store.connect(db_file)
    store.init_schema(conn)
    return conn


def _init_state() -> None:
    import streamlit as st

    st.session_state.setdefault(KEY_EV_SECTION, "Events")
    st.session_state.setdefault(KEY_EV_STATE, "All")
    st.session_state.setdefault(KEY_EV_SEVERITY, "All")
    st.session_state.setdefault(KEY_EV_PAGE, 0)
    st.session_state.setdefault(KEY_EV_SELECTED, "")
    st.session_state.setdefault(KEY_AL_STATUS, "open")
    st.session_state.setdefault(KEY_AL_TYPE, "All")
    st.session_state.setdefault(KEY_AL_PAGE, 0)
    st.session_state.setdefault(KEY_AL_SELECTED, "")


# --- Events ---

def _render_events(conn) -> None:
    import streamlit as st

    total = store.count_events(conn)
    st.caption(f"{total:,} activity events (one per hotspot cluster).")
    st.caption("This list is global (all dates); Map, Analytics, "
               "Pipeline, and Exports follow the shared sidebar range.")
    if total == 0:
        st.info("No events detected in the configured events database "
                "(0 event records registered — verified empty event store, not unavailable data).")
    col_a, col_b = st.columns(2)
    with col_a:
        st.selectbox("Lifecycle state", options=["All", *model.LIFECYCLE_STATES],
                     key=KEY_EV_STATE)
    with col_b:
        st.selectbox("Severity", options=["All", *model.SEVERITY_LEVELS],
                     key=KEY_EV_SEVERITY)
    state = st.session_state.get(KEY_EV_STATE)
    severity = st.session_state.get(KEY_EV_SEVERITY)
    page = max(0, int(st.session_state.get(KEY_EV_PAGE, 0)))
    events = store.list_events(
        conn, state=None if state == "All" else state,
        severity=None if severity == "All" else severity,
        limit=PAGE_SIZE, offset=page * PAGE_SIZE)
    if total > 0 and not events:
        st.info("No events detected matching the active filters.")
    st.dataframe(frames.events_frame(events), width="stretch")
    nav_a, nav_b = st.columns(2)
    with nav_a:
        if st.button("Previous page", key="cal_ev_prev",
                     disabled=page == 0):
            st.session_state[KEY_EV_PAGE] = page - 1
    with nav_b:
        if st.button("Next page", key="cal_ev_next",
                     disabled=len(events) < PAGE_SIZE):
            st.session_state[KEY_EV_PAGE] = page + 1
    picked = st.text_input("Inspect event", key=KEY_EV_SELECTED,
                           placeholder="e.g. ev-cl0000161-a1 or cl0000161")
    if picked:
        clean = picked.strip()
        if clean.startswith("cl"):
            # Map-tab cross-reference: resolve a cluster id to its event.
            from ..events.association import event_id_for

            try:
                clean = event_id_for(clean)
            except ValueError:
                event = None
            else:
                event = store.get_event(conn, clean)
        else:
            event = store.get_event(conn, clean)
        if event is None:
            st.warning(f"No event with id {picked.strip()!r}.")
        else:
            _render_event_detail(conn, event)


def _render_event_detail(conn, event: dict) -> None:
    import streamlit as st

    st.write(f"### Event {event['event_id']}")
    cols = st.columns(4)
    cols[0].metric("State", event["lifecycle_state"])
    cols[1].metric("Severity", event["severity_name"])
    cols[2].metric("Observations", f"{event['detection_count']:,}")
    frp_max = (event.get("frp") or {}).get("frp_max")
    cols[3].metric("Max FRP", "unavailable" if frp_max is None
                   else f"{frp_max} MW")
    st.write(f"Observed {event['first_observed_utc'][:10]} to "
             f"{event['last_observed_utc'][:10]} | MODIS "
             f"{event['sensor_modis_n']} / VIIRS {event['sensor_viirs_n']} | "
             f"satellites {', '.join(event.get('satellites') or []) or '—'}")
    st.write(f"Source clusters: {', '.join(event.get('cluster_ids') or [])}")
    with st.expander("Metrics and evidence"):
        st.write({"activity": event.get("activity"), "frp": event.get("frp"),
                  "bbox": event.get("bbox"),
                  "centroid": event.get("centroid"),
                  "completeness": event.get("completeness"),
                  "ambiguity": event.get("ambiguity")})
        st.caption(event.get("severity_provenance", ""))
    with st.expander("Provenance"):
        st.write(event.get("provenance"))
        st.caption("Detection-level IDs resolve on demand via the pinned "
                   "Phase 2C record (see detection_resolution).")
        st.write(event.get("detection_resolution"))
    with st.expander("Lifecycle history"):
        history = store.list_transitions(conn, event["event_id"])
        if history:
            st.dataframe(frames.lifecycle_frame(history),
                         width="stretch")
        else:
            st.caption("No transitions yet (seed state).")
        st.caption(model.RESOLVED_DISCLAIMER)
    with st.expander("Human review"):
        reviews = store.list_reviews(conn, event["event_id"])
        if reviews:
            st.dataframe(frames.reviews_frame(reviews),
                         width="stretch")
        else:
            st.caption("No reviews recorded.")
        if event.get("review_required"):
            st.warning("Human review is required before this event can "
                       "resolve (High/Critical severity).")
        reviewer = st.text_input("Reviewer label (optional)",
                                 key="cal_ev_reviewer",
                                 placeholder="e.g. duty-analyst")
        decision = st.selectbox("Decision", options=["approved", "rejected"],
                                key="cal_ev_decision")
        note = st.text_input("Note (optional)", key="cal_ev_note")
        if st.button("Record review", key="cal_ev_review"):
            store.record_review(conn, event_id=event["event_id"],
                                reviewer_label=reviewer, decision=decision,
                                note=note)
            st.success(f"Review recorded ({decision}).")
    with st.expander("Linked alerts"):
        linked = store.list_alerts(conn, event_id=event["event_id"],
                                   limit=100)
        if linked:
            st.dataframe(frames.linked_alerts_frame(linked),
                         width="stretch")
        else:
            st.caption("No alerts are attached to this event.")
    with st.expander("Event report"):
        history = store.list_transitions(conn, event["event_id"])
        chain = {}
        chain.update(query.exec_chain())
        chain.update(query.phase4_chain())
        chain.update(query.phase5_chain())
        report = event_report.event_report_markdown(
            event, transitions=history, alerts=linked, exec_chain=chain)
        st.download_button("Download event report (markdown)",
                           data=report.encode("utf-8"),
                           file_name=f"fireagent-event-{event['event_id']}.md",
                           mime="text/markdown",
                           key=f"cal_evr_dl_{event['event_id']}",
                           help="Methods, figures, provenance, and "
                                "limitations for this event.")


# --- Alerts ---

def _render_alerts(conn) -> None:
    import streamlit as st

    total = store.count_alerts(conn)
    st.caption(f"{total:,} alerts. {model.SEVERITY_SCALE_NOTE}")
    col_a, col_b = st.columns(2)
    with col_a:
        st.selectbox("Status", options=["open", "acknowledged", "resolved"],
                     key=KEY_AL_STATUS)
    with col_b:
        st.selectbox("Type", options=["All", *model.ALERT_TYPES],
                     key=KEY_AL_TYPE)
    status = st.session_state.get(KEY_AL_STATUS, "open")
    alert_type = st.session_state.get(KEY_AL_TYPE, "All")
    page = max(0, int(st.session_state.get(KEY_AL_PAGE, 0)))
    found = store.list_alerts(
        conn, status=status,
        alert_type=None if alert_type == "All" else alert_type,
        limit=PAGE_SIZE, offset=page * PAGE_SIZE)
    st.dataframe(frames.alerts_list_frame(found), width="stretch")
    nav_a, nav_b = st.columns(2)
    with nav_a:
        if st.button("Previous page", key="cal_al_prev",
                     disabled=page == 0):
            st.session_state[KEY_AL_PAGE] = page - 1
    with nav_b:
        if st.button("Next page", key="cal_al_next",
                     disabled=len(found) < PAGE_SIZE):
            st.session_state[KEY_AL_PAGE] = page + 1
    dq = store.list_alerts(conn, status="open", alert_type="Data Quality",
                           limit=10)
    if dq:
        st.warning(f"{len(dq)} open data-quality notice(s): availability "
                   "gaps affect interpretation (see list above).")
    else:
        st.caption("No open data-quality notices.")
    picked = st.text_input("Inspect alert", key=KEY_AL_SELECTED,
                           placeholder="e.g. alt-asp-ev-cl0000161-a1-r1")
    if picked:
        alert = store.get_alert(conn, picked.strip())
        if alert is None:
            st.warning(f"No alert with id {picked.strip()!r}.")
        else:
            _render_alert_detail(conn, alert)


def _render_alert_detail(conn, alert: dict) -> None:
    import streamlit as st

    st.write(f"### Alert {alert['alert_id']}")
    cols = st.columns(4)
    cols[0].metric("Type", alert["alert_type"])
    cols[1].metric("Severity", alert["severity_name"])
    cols[2].metric("Status", alert["status"])
    cols[3].metric("Period", alert["period"] or "—")
    st.write(f"Scope {alert['scope']} | metric {alert['metric']} | event "
             f"{alert['event_id'] or 'pilot-wide (no single event implicated)'}")
    if alert["alert_type"] in ("Activity Spike", "High FRP"):
        st.caption("Retrospective-only classification: severity bands were "
                   "derived from the full available cluster population, "
                   "including observations later than this alert's period. "
                   "Not validated as a real-time alert signal.")
    elif alert["alert_type"] in ("Historical Anomaly", "Cluster Anomaly"):
        st.caption("Phase 4 trailing-reference classification: the reference "
                   "period strictly precedes the evaluation period.")
    with st.expander("Explanation", expanded=True):
        st.text(alert.get("explanation", "unavailable"))
    with st.expander("Trigger and references"):
        st.write({"trigger": alert.get("trigger"),
                  "baseline_ref": alert.get("baseline_ref"),
                  "data_quality": alert.get("data_quality"),
                  "rule_version": alert.get("rule_version")})
    with st.expander("Alert history"):
        history = store.list_alert_history(conn, alert["alert_id"])
        st.dataframe(frames.alert_history_frame(history),
                     width="stretch")
    note = st.text_input("Note (optional)", key="cal_al_note")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Acknowledge", key="cal_al_ack",
                     disabled=alert["status"] != "open"):
            store.set_alert_status(conn, alert["alert_id"], "acknowledged",
                                   note=note)
            st.success("Acknowledged.")
    with col_b:
        if st.button("Resolve", key="cal_al_resolve",
                     disabled=alert["status"] == "resolved"):
            store.set_alert_status(conn, alert["alert_id"], "resolved",
                                   note=note)
            st.success("Resolved.")


# --- Notifications ---

def _render_notifications(conn) -> None:
    import streamlit as st

    status = notify.smtp_status()
    if status["available"]:
        st.success(f"Email delivery available ({status['reason']}).")
    else:
        st.warning(f"Email delivery unavailable: {status['reason']}. The "
                   "application works normally; notifications are recorded "
                   "as skipped, never as sent.")
    st.write("#### Subscribe")
    email = st.text_input("Email address", key="cal_ntf_email",
                          placeholder="you@example.com")
    if st.button("Subscribe", key="cal_ntf_sub"):
        try:
            result = notify.subscribe(conn, email)
        except ValueError as exc:
            st.error(str(exc))
        else:
            if result["status"] == "confirmed":
                st.info("This address is already confirmed.")
            elif status["available"]:
                subject, body = notify.confirmation_email(
                    result["sub_id"], result["confirm_token"])
                if notify.send_via_smtp(email.strip().lower(), subject,
                                        body):
                    st.success("Confirmation email sent. Paste the token "
                               "below to confirm.")
                else:
                    st.error("Confirmation email could not be sent; the "
                             "subscription stays pending.")
            else:
                st.info("Subscription recorded as pending. Confirming "
                        "requires email delivery (SMTP is not configured).")
    st.write("#### Confirm / manage / unsubscribe")
    token = st.text_input("Paste confirmation or management token",
                          key="cal_ntf_token", type="password")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Confirm subscription", key="cal_ntf_confirm"):
            result = notify.confirm(conn, token)
            if result["status"] == "confirmed":
                st.success("Subscription confirmed.")
            else:
                st.error(result.get("reason", result["status"]))
    with col_b:
        if st.button("Unsubscribe", key="cal_ntf_unsub"):
            result = notify.unsubscribe(conn, token)
            if result["status"] == "unsubscribed":
                st.success("Unsubscribed.")
            else:
                st.error(result.get("reason", result["status"]))
    with st.expander("Delivery history (latest)"):
        rows = store.list_notifications(conn, limit=50)
        if rows:
            st.dataframe(frames.notifications_frame(rows),
                         width="stretch")
            st.caption("Subscriber addresses are never displayed.")
        else:
            st.caption("No notifications recorded yet.")


# --- Qwen Intelligence ---

QWEN_EVENT_OPTIONS_CAP = 100


def _events_in_range(conn, start: str, end: str,
                     limit: int = QWEN_EVENT_OPTIONS_CAP):
    """Real events overlapping [start, end]: ([{id, label}], truncated).

    Resolves date-range clusters to their activity events; exact
    internal ids preserved. Empty list (never invented ids) when the
    range has no resolvable events.
    """
    from ..events.association import event_id_for

    try:
        clusters = query.query_clusters(
            date_start=start, date_end=end,
            limit=limit + 1).to_dict("records")
    except (ValueError, KeyError, FileNotFoundError):
        return [], False
    options = []
    for cluster in clusters[:limit]:
        try:
            eid = event_id_for(cluster["cluster_id"])
        except ValueError:
            continue
        event = store.get_event(conn, eid)
        if event is None:
            continue
        first = str(event.get("first_observed_utc") or "")[:10]
        last = str(event.get("last_observed_utc") or "")[:10]
        options.append({
            "id": eid,
            "label": (f"{eid} | {first}→{last} | "
                      f"{event.get('detection_count', '?')} obs "
                      f"(M{event.get('sensor_modis_n', '?')}/"
                      f"V{event.get('sensor_viirs_n', '?')}) | "
                      f"{event.get('lifecycle_state', '?')}/"
                      f"{event.get('severity_name', '?')}")})
    return options, len(clusters) > limit


def _render_operator_summary(result) -> None:
    import streamlit as st

    agg = (result.operator or {}).get("aggregate") or {}
    by_sensor = agg.get("by_sensor") or {}
    if by_sensor:
        st.dataframe(frames.operator_sensor_frame(by_sensor),
                     width="stretch")
    regimes = agg.get("regime_histogram") or {}
    if regimes:
        st.caption("Detectability regimes (diagnostic ±2/r bands): "
                   + ", ".join(f"{label} {count:,}"
                               for label, count in sorted(regimes.items())))
    l2_cov = agg.get("l2_coverage") or {}
    st.caption(f"Level-2 extracts: AOD on "
               f"{l2_cov.get('with_aod', 0)} row(s), cloud on "
               f"{l2_cov.get('with_cloud', 0)} row(s) "
               f"(null = unavailable, never zero).")
    st.caption("Equations: pixel Eq.5/6/7 experimental "
               "(det-frp-oop-v1); gridded Eq.8–11 and AOD Eq.12 not "
               "implemented; cross-sensor mapping experimental.")


def _render_qwen(conn) -> None:
    import streamlit as st

    from ..events import models as models_mod

    config = qwen.QwenConfig.from_settings()
    status = config.status()
    if status["available"]:
        st.success(f"Qwen interpretation available ({status['reason']}).")
    else:
        st.warning(f"Qwen interpretation unavailable: {status['reason']}. "
                   "Deterministic explanations are shown instead. "
                   "Scientific data and metrics are unaffected.")
    with st.expander("Model discovery (OpenRouter catalog)"):
        st.caption("Discovers Qwen chat models available to the configured "
                   "API key (catalog + key check only — never a paid "
                   "completion). Paid models need explicit approval below.")
        record = models_mod.load_discovery(conn)
        if record is None:
            st.caption("No fresh discovery on record (24 h cache).")
        else:
            selection = record.get("selection") or {}
            st.write("Catalog:", f"{record.get('catalog_count', 0)} models, "
                     f"fetched {str(record.get('fetched_at'))[:19]} UTC")
            st.write("Key:", "authorized"
                     if record.get("key_authorized")
                     else f"not verified ({record.get('key_label')})")
            if selection.get("model_id"):
                st.write("Selected:", selection["model_id"],
                         f"({selection.get('price_class')})")
            st.caption(selection.get("reason", ""))
            if record.get("error"):
                st.warning(record["error"])
            if selection.get("needs_approval"):
                st.warning("The cheapest available Qwen model is paid. "
                           "Automatic use is disabled.")
                if st.button("Approve paid model "
                             f"{selection.get('model_id')}",
                             key="cal_qw_approve"):
                    models_mod.approve_paid_model(
                        conn, selection["model_id"])
                    st.success("Approval recorded. Refresh discovery to "
                               "apply it.")
        if st.button("Refresh model catalog", key="cal_qw_refresh"):
            with st.spinner("Querying model catalog…"):
                try:
                    fresh = models_mod.refresh_discovery(
                        conn, api_key=config.api_key,
                        timeout_s=config.timeout_s)
                except Exception as exc:
                    st.error(f"Discovery failed: {exc}")
                else:
                    if fresh.get("error"):
                        st.warning(f"Discovery issue: {fresh['error']}")
                    else:
                        st.success(
                            f"Catalog refreshed: "
                            f"{fresh.get('catalog_count', 0)} models.")
    effective, why = models_mod.effective_model(conn, config)
    if effective:
        st.caption(f"Explain will use: {effective} ({why}).")
    else:
        st.caption(f"No model selected ({why}); Explain uses the "
                   f"deterministic fallback.")
    options, truncated = _events_in_range(
        conn,
        st.session_state.get("cal_range_start", "2023-01-01"),
        st.session_state.get("cal_range_end", "2023-12-31"))
    labels = {item["id"]: item["label"] for item in options}
    choices = [item["id"] for item in options]
    preset = st.session_state.get("cal_qw_target", "")
    if preset and preset not in choices:
        choices = [*choices, preset]  # preserve programmatic selection
        labels[preset] = preset
    if not choices:
        choices = ["(no events in this date range)"]
        labels = {choices[0]: choices[0]}
    target = st.selectbox(
        "Event (from the shared date range)", options=choices,
        key="cal_qw_target",
        format_func=lambda item: labels.get(item, item))
    if truncated:
        st.caption("Event list truncated to 100 (narrow the date range).")
    with st.expander("Or explain an alert id"):
        alert_target = st.text_input(
            "Alert id", key="cal_qw_alert",
            placeholder="e.g. alt-asp-ev-cl0000161-a1-r1")
    if st.button("Explain", key="cal_qw_go") and (
            target or alert_target):
        import dataclasses

        # Discovered (or explicit) model override; validation, fallback,
        # and Explain-only gating inside qwen.explain are unchanged.
        run_config = dataclasses.replace(config, model=effective or "")
        picked = (alert_target or "").strip() or (target or "").strip()
        clean = picked
        event = store.get_event(conn, clean)
        alert = store.get_alert(conn, clean) if event is None else None
        if event is not None:
            linked = store.list_alerts(conn, event_id=event["event_id"],
                                       limit=50)
            payload = qwen.build_event_payload(event, linked)
            result = qwen.explain("event", event["event_id"], payload,
                                  config=run_config, conn=conn)
        elif alert is not None:
            event = store.get_event(conn, alert["event_id"]) \
                if alert.get("event_id") else None
            payload = qwen.build_alert_payload(alert, event)
            result = qwen.explain("alert", alert["alert_id"], payload,
                                  config=run_config, conn=conn)
        else:
            st.warning(f"No event or alert with id {clean!r}.")
            return
        st.write(f"#### {result['label']}")
        st.text(result["text"])
        st.caption("Evidence references: "
                   + (", ".join(result["evidence_ids"])
                      or "none recorded"))
        if result["status"] != "ok":
            st.caption(f"Qwen status: {result['status']} "
                       f"({result['validation']}).")
        elif result.get("latency_ms") is not None:
            st.caption(f"Model {result['model']} · "
                       f"{result['latency_ms']:.0f} ms · validated against "
                       "supplied evidence.")
        st.caption("AI text is an interpretation aid, never an official "
                   "assessment or authoritative fire report.")
        with st.expander("Validation receipt"):
            st.caption("Guardrails applied to every Qwen answer "
                       "(failures fall back, never display).")
            for item in qwen.describe_checks():
                st.write(f"- **{item['check']}**: {item['rule']}")
            st.write("This explanation:", result["status"], "—",
                     result["validation"])
            if result["status"] != "ok":
                st.caption("Deterministic template: values copied from the "
                           "evidence payload; no model inference ran.")
    st.divider()
    st.caption("For the full Start → End automated analysis with agent "
               "trace and CSV exports, open the Pipeline section above.")


# --- Custom Alerts (user-defined, manual evaluation only) ---

KEY_CR_EDIT = "cal_cr_edit"
KEY_CR_DELETE_ARM = "cal_cr_delete_arm"


def _custom_caption() -> None:
    import streamlit as st

    st.caption("User-defined threshold records over stored metrics and "
               "anomaly classifications. Evaluation is manual (press a Run "
               "button) — no background scheduler exists. A threshold "
               "crossing is not proof of a wildfire; custom records never "
               "enter the system alerts table.")


def _rule_summary(rule: dict) -> str:
    parts = [f"{rule['metric']}/{rule['scope']} {rule['grain']} "
             f"{rule['period']}"]
    if rule.get("operator"):
        from ..events.custom import OPERATOR_LABELS

        parts.append(f"value {OPERATOR_LABELS[rule['operator']]} "
                     f"{rule['threshold']}")
    if rule.get("anomaly_class"):
        parts.append(f"class is {rule['anomaly_class']}")
    return "; ".join(parts)


def _render_custom(conn, metrics, anomalies) -> None:
    import streamlit as st

    from ..events import custom as custom_mod

    _custom_caption()
    if metrics is None and query.artifact_status("2D", query.METRICS_FILENAME)["available"]:
        with st.spinner("Loading metrics…"):
            try:
                metrics = query.load_metrics()
            except FileNotFoundError:
                metrics = None
    if anomalies is None and query.artifact_status("4C", "anomalies_4c.json")["available"]:
        with st.spinner("Loading anomalies…"):
            try:
                anomalies = query.load_anomalies()
            except FileNotFoundError:
                anomalies = None
    if metrics is None or anomalies is None:
        st.warning("Custom rule evaluation requires Phase 2D (metrics_2d.json) and "
                   "Phase 4C (anomalies_4c.json) outputs, which are unavailable in this deployment.")
    rules = store.list_custom_rules(conn)
    if rules:
        st.dataframe(frames.custom_rules_frame(
            rules, [_rule_summary(r) for r in rules]),
            width="stretch")
    else:
        st.caption("No custom rules yet. Create one below.")

    editing = None
    edit_id = st.session_state.get(KEY_CR_EDIT)
    if edit_id:
        editing = store.get_custom_rule(conn, edit_id)
        if editing is None:
            st.session_state[KEY_CR_EDIT] = None
    with st.form("cal_cr_form"):
        st.write("### Edit rule" if editing else "### Create rule")
        name = st.text_input("Name", value=(editing or {}).get("name", ""),
                             max_chars=120, key="cal_cr_name")
        description = st.text_input(
            "Description (optional)",
            value=(editing or {}).get("description", ""), key="cal_cr_desc")
        col_a, col_b = st.columns(2)
        with col_a:
            metric = st.selectbox("Metric", key="cal_cr_metric",
                                  options=["detection", "cluster", "frp"],
                                  index=["detection", "cluster",
                                         "frp"].index(
                                      (editing or {}).get("metric",
                                                          "detection")))
            grain = st.selectbox("Grain", key="cal_cr_grain", options=["daily", "weekly",
                                                   "monthly"],
                                 index=["daily", "weekly", "monthly"].index(
                                     (editing or {}).get("grain", "daily")))
            operator = st.selectbox(
                "Operator (or 'no threshold')", key="cal_cr_op",
                options=["no threshold", "gt", "gte", "lt", "lte"],
                index=["no threshold", "gt", "gte", "lt", "lte"].index(
                    (editing or {}).get("operator") or "no threshold"),
                format_func=lambda o: {"no threshold": "No threshold",
                                       "gt": ">", "gte": ">=", "lt": "<",
                                       "lte": "<="}[o])
        with col_b:
            scope = st.text_input(
                "Scope (exact series scope, e.g. raw, modis, Terra, "
                "harmonized, count)", key="cal_cr_scope",
                value=(editing or {}).get("scope", "raw"))
            period = st.text_input(
                "Period (YYYY-MM-DD, YYYY-Www, or YYYY-MM)", key="cal_cr_period",
                value=(editing or {}).get("period", "2023-08-15"))
            threshold = st.text_input(
                "Threshold (number; empty disables the threshold)", key="cal_cr_threshold",
                value="" if (editing or {}).get("threshold") is None
                else str(editing["threshold"]))
        anomaly_class = st.selectbox(
            "Anomaly class (optional; sensor_aware, leakage-checked)", key="cal_cr_anomaly",
            options=["none", *custom_mod.ANOMALY_CLASSES],
            index=0 if not (editing or {}).get("anomaly_class") else
            (["none", *custom_mod.ANOMALY_CLASSES].index(
                editing["anomaly_class"])))
        col_c, col_d = st.columns(2)
        with col_c:
            severity = st.selectbox(
                "Severity label", key="cal_cr_severity", options=list(model.SEVERITY_LEVELS),
                index=list(model.SEVERITY_LEVELS).index(
                    (editing or {}).get("severity_name", "Moderate")))
        with col_d:
            notify_flag = st.checkbox(
                "Request email on trigger", key="cal_cr_notify", value=bool(
                    (editing or {}).get("notify", False)),
                help="Queues a notification intent; delivery needs SMTP plus "
                     "an explicit send action (no scheduler).")
        submitted = st.form_submit_button(
            "Save changes" if editing else "Save rule")
    if submitted:
        try:
            record = custom_mod.validate_rule({
                "rule_id": (editing or {}).get("rule_id"),
                "name": name, "description": description, "metric": metric,
                "scope": scope.strip(), "grain": grain,
                "period": period.strip(),
                "operator": None if operator == "no threshold" else operator,
                "threshold": threshold.strip() or None,
                "anomaly_class": None if anomaly_class == "none"
                else anomaly_class,
                "severity_name": severity, "notify": notify_flag,
                "enabled": (editing or {}).get("enabled", True),
            })
        except ValueError as exc:
            st.error(f"Invalid rule: {exc}")
        else:
            now = custom_mod._utcnow()
            existing = (store.get_custom_rule(conn, record["rule_id"])
                        if editing else None)
            record["created_at"] = (existing or {}).get("created_at", now)
            record["updated_at"] = now
            store.upsert_custom_rule(conn, record)
            st.session_state[KEY_CR_EDIT] = None
            st.success(f"Rule saved ({record['rule_id']}).")
    if editing and st.button("Cancel edit", key="cal_cr_cancel"):
        st.session_state[KEY_CR_EDIT] = None

    for rule in store.list_custom_rules(conn):
        with st.expander(f"{rule['name']} ({rule['rule_id']})"):
            st.write("Condition:", _rule_summary(rule))
            st.write("Severity:", rule["severity_name"], "| Enabled:",
                     bool(rule["enabled"]), "| Notify on trigger:",
                     bool(rule["notify"]), "| Geography:",
                     rule["geo_scope"], "(pilot-wide; no regional metrics "
                     "exist, so no narrower AOI is offered)")
            if rule["description"]:
                st.caption(rule["description"])
            btn_a, btn_b, btn_c, btn_d = st.columns(4)
            with btn_a:
                if st.button("Run now", key=f"cal_cr_run_{rule['rule_id']}"):
                    if metrics is None or anomalies is None:
                        st.warning("Cannot evaluate rule: required analytical outputs "
                                   "(CAL_EXEC_2D / CAL_EXEC_4C) are unavailable.")
                    else:
                        try:
                            outcome = custom_mod.run_evaluation(
                                conn, rule["rule_id"], metrics=metrics,
                                anomalies=anomalies)
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            st.success(f"Triggered: {outcome['reason']}"
                                       if outcome["triggered"] else
                                       f"Not triggered: {outcome['reason']}")
            with btn_b:
                label = "Disable" if rule["enabled"] else "Enable"
                if st.button(label, key=f"cal_cr_toggle_{rule['rule_id']}"):
                    store.set_custom_rule_enabled(conn, rule["rule_id"],
                                                  not rule["enabled"])
            with btn_c:
                if st.button("Edit", key=f"cal_cr_edit_{rule['rule_id']}"):
                    st.session_state[KEY_CR_EDIT] = rule["rule_id"]
            with btn_d:
                armed = st.session_state.get(KEY_CR_DELETE_ARM)
                if armed != rule["rule_id"]:
                    if st.button("Delete",
                                 key=f"cal_cr_del_{rule['rule_id']}"):
                        st.session_state[KEY_CR_DELETE_ARM] = rule["rule_id"]
                else:
                    if st.button("Confirm delete",
                                 key=f"cal_cr_delc_{rule['rule_id']}"):
                        store.delete_custom_rule(conn, rule["rule_id"])
                        st.session_state[KEY_CR_DELETE_ARM] = None
                        st.success("Rule deleted (hits and evals kept).")
            evals = store.list_custom_evals(conn, rule["rule_id"], limit=20)
            if evals:
                st.dataframe(frames.custom_evals_frame(evals),
                             width="stretch")

    st.write("### Triggered records")
    hits = store.list_custom_hits(conn, limit=100)
    if hits:
        st.dataframe(frames.custom_hits_frame(hits), width="stretch")
        hit_id = st.text_input("Acknowledge hit id", key="cal_cr_hit_ack",
                               placeholder="crh-…")
        if st.button("Acknowledge", key="cal_cr_ack") and hit_id.strip():
            try:
                store.set_custom_hit_status(conn, hit_id.strip(),
                                            "acknowledged")
                st.success("Acknowledged.")
            except ValueError as exc:
                st.error(str(exc))
    else:
        st.caption("No triggered records yet.")

    st.write("### Pending notifications")
    status = notify.smtp_status()
    queued = store.list_custom_notifications(conn, status="queued")
    if not status["available"]:
        st.warning(f"SMTP unavailable ({status['reason']}): intents stay "
                   f"queued ({len(queued)}). Nothing is sent.")
    elif not queued:
        st.caption("SMTP ready. No queued custom notifications.")
    else:
        st.caption(f"{len(queued)} queued custom notification(s).")
        if st.button("Send pending notifications", key="cal_cr_send"):
            stats = custom_mod.send_pending_custom(conn,
                                                   sender=notify.send_via_smtp)
            st.success(f"Sent {stats['sent']}, failed {stats['failed']}, "
                       f"skipped {stats['skipped']}."
                       + (f" {stats['reason']}" if stats.get("reason")
                          else ""))


# --- Tab ---

def _render_pipeline(conn) -> None:
    """Integrated Start → End automated analysis (no agent config)."""
    import dataclasses

    import streamlit as st

    from ..events import models as models_mod
    from ..pipeline import runner as pipe_runner

    st.write("### Analysis pipeline")
    st.caption("Pick a date range and run: acquisition → detection "
               "intelligence → event analysis → observation operator → "
               "validation → CSV exports. No agent or event-type "
               "configuration needed.")
    from . import aoi as aoi_mod
    from . import filters as filters_mod

    filters_mod.sync_inputs_from_range(
        st.session_state, start_key="cal_pipe_start",
        end_key="cal_pipe_end", sentinel_key="cal_pipe_synced_range")
    try:
        pipe_bbox = aoi_mod.resolve_bbox(
            mode=st.session_state.get(filters_mod.KEY_AOI_MODE, "full"),
            region_id=st.session_state.get(filters_mod.KEY_AOI_REGION),
            custom=(st.session_state.get(filters_mod.KEY_AOI_W),
                    st.session_state.get(filters_mod.KEY_AOI_S),
                    st.session_state.get(filters_mod.KEY_AOI_E),
                    st.session_state.get(filters_mod.KEY_AOI_N)),
            viewport=st.session_state.get("cal_map_bbox"),
            saved=None)
    except aoi_mod.AoIError:
        pipe_bbox = None
    if pipe_bbox:
        st.caption(f"Geographic filter active: lon {pipe_bbox[0]:.2f}…"
                   f"{pipe_bbox[2]:.2f}, lat {pipe_bbox[1]:.2f}…"
                   f"{pipe_bbox[3]:.2f} (sidebar Area of interest).")
    else:
        st.caption("Geographic filter: full pilot (USA). Set a sidebar "
                   "Area of interest to restrict the run.")
    st.caption("Date inputs follow the shared sidebar range until you "
               "edit them here.")
    col_a, col_b = st.columns(2)
    with col_a:
        start = st.text_input("Start Date", value="2023-08-15",
                              key="cal_pipe_start",
                              placeholder="YYYY-MM-DD")
    with col_b:
        end = st.text_input("End Date", value="2023-08-16",
                            key="cal_pipe_end", placeholder="YYYY-MM-DD")
    l2_match = st.checkbox(
        "Match Level-2 granules (needs network + Earthdata; "
        "ranges of 7 days or fewer)",
        value=False, key="cal_pipe_l2")
    qcfg = qwen.QwenConfig.from_settings()
    eff_model, eff_reason = models_mod.effective_model(conn, qcfg)
    if eff_model:
        st.caption(f"Qwen model: {eff_model} ({eff_reason}).")
    else:
        st.caption(f"Qwen unavailable: {eff_reason}.")
    granite = GraniteProvider()
    granite_ok, granite_reason = granite.available()
    if granite_ok:
        st.caption(f"IBM Granite fallback ready ({granite.model}).")
    else:
        st.caption(f"IBM Granite unavailable: {granite_reason}. "
                   "Deterministic fallback covers all agents.")
    if st.button("Run Analysis", key="cal_pipe_go"):
        providers_list = []
        if eff_model:
            providers_list.append(QwenProvider(
                dataclasses.replace(qcfg, model=eff_model)))
        providers_list.append(GraniteProvider())
        chain = ProviderChain(providers_list)
        box = st.status("Running pipeline…", expanded=True)

        def _progress(name: str, state: str, detail: str) -> None:
            box.write(f"{name}: {state}"
                      + (f" — {detail}" if detail else ""))

        try:
            fresh = pipe_runner.run_pipeline(
                (start or "").strip(), (end or "").strip(),
                events_conn=conn, chain=chain, progress=_progress,
                l2_match=bool(l2_match), bbox=pipe_bbox)
        except ValueError as exc:
            box.update(label="Pipeline rejected the input", state="error")
            st.error(str(exc))
            return
        box.update(label=f"Pipeline finished ({fresh.execution_id})",
                   state="complete", expanded=False)
        # Persist the completed run: downloads and reruns reuse these
        # bytes without recomputing acquisition or agent calls.
        st.session_state["cal_pipe_result"] = fresh
        st.session_state["cal_pipe_result_range"] = (
            fresh.date_start, fresh.date_end)
    result = st.session_state.get("cal_pipe_result")
    if result is None:
        st.caption("No pipeline run yet in this session.")
        return
    shown = st.session_state.get("cal_pipe_result_range", ("?", "?"))
    if (start, end) != (shown[0], shown[1]):
        st.caption(f"Inputs changed since execution {result.execution_id} "
                   f"(ran {shown[0]} to {shown[1]}). Run again to "
                   f"refresh; downloads below are the stored run.")
    else:
        st.caption(f"Execution {result.execution_id} "
                   f"({result.date_start} to {result.date_end}).")
    st.write("#### Stages")
    st.dataframe(frames.stages_frame(result.stages), width="stretch")
    st.write("#### Agent executions")
    if result.agent_runs:
        st.dataframe(frames.agent_runs_frame(result.agent_runs),
                     width="stretch")
        for run in result.agent_runs:
            if run["status"] != "ok":
                st.caption(f"Fallback ({run['agent']} → "
                           f"{run['target_id']}): "
                           f"{run['fallback_reason']}")
    else:
        st.caption("No agent executions in this run.")
    with st.expander("Validation checks"):
        for check in result.validation.get("checks", []):
            mark = "PASS" if check["passed"] else "FAIL"
            st.write(f"- {mark}: {check['check']}"
                     + (f" ({check['detail']})" if check.get("detail")
                        else ""))
    with st.expander("Metrics"):
        st.dataframe(frames.metrics_frame(result.metrics),
                     width="stretch")
    with st.expander("Observation operator"):
        _render_operator_summary(result)
    if result.limitations:
        st.caption("Limitations: " + " ".join(result.limitations))
    st.write("#### Downloads")
    if result.csvs:
        for kind in ("observations", "events", "agents", "metrics"):
            payload = result.csvs.get(kind)
            if payload is None:
                st.caption(f"{kind} export unavailable for this run.")
                continue
            st.download_button(
                f"Download {kind} CSV",
                data=payload.encode("utf-8"),
                file_name=f"fireagent-{kind}-{result.execution_id}.csv",
                mime="text/csv", key=f"cal_pipe_dl_{kind}")
    else:
        st.caption("No exports were generated (see stage statuses).")


def render_events_tab(metrics=None, anomalies=None) -> None:
    """Full Events & Alerts tab (sections + methodology note)."""
    import streamlit as st

    _init_state()
    ev_st = query.events_db_status()
    if not ev_st["available"]:
        clu_st = query.artifact_status("2C", "clusters.parquet")
        met_st = query.artifact_status("2D", query.METRICS_FILENAME)
        anom_st = query.artifact_status("4C", "anomalies_4c.json")
        st.warning(
            f"Event data unavailable in this deployment ({ev_st['status']}: {ev_st['reason']}). "
            "This represents unavailable event data, not zero detected events."
        )
        st.info(
            "The events database has not been built yet or is not configured in this deployment.\n\n"
            f"- Event store (`{ev_st['env_var']}`): `{ev_st['path'] or '(unconfigured)'}` "
            f"(`{ev_st['status']}`)\n"
            f"- Stage 2C clusters (`{clu_st['env_var']}`): `{clu_st['status']}` "
            f"(`{clu_st['path'] or 'clusters.parquet'}`)\n"
            f"- Stage 2D metrics (`{met_st['env_var']}`): `{met_st['status']}` "
            f"(`{met_st['path'] or query.METRICS_FILENAME}`)\n"
            f"- Stage 4C anomalies (`{anom_st['env_var']}`): `{anom_st['status']}` "
            f"(`{anom_st['path'] or 'anomalies_4c.json'}`)\n\n"
            f"How to enable: {ev_st['remediation']} "
            "Calendar metrics remain unaffected when Phase 2D outputs are present."
        )
        st.info(METHODOLOGY_NOTE)
        return
    conn = _connect()
    try:
        section = st.radio("Section", options=["Events", "Alerts",
                                               "Pipeline",
                                               "Notifications",
                                               "Qwen Intelligence",
                                               "Custom Alerts"],
                           key=KEY_EV_SECTION, horizontal=True)
        if section == "Events":
            _render_events(conn)
        elif section == "Alerts":
            _render_alerts(conn)
        elif section == "Pipeline":
            _render_pipeline(conn)
        elif section == "Notifications":
            _render_notifications(conn)
        elif section == "Qwen Intelligence":
            _render_qwen(conn)
        else:
            _render_custom(conn, metrics, anomalies)
    finally:
        conn.close()
    st.info(METHODOLOGY_NOTE)
