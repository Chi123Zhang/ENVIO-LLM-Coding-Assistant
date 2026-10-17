from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()


def test_original_page_and_orcca_demo_render_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ORCCA_DEPLOYMENT_MODE", raising=False)
    at = app()
    assert not at.exception
    assert at.title[0].value == "ENVIO LLM Coding Assistant"
    assert at.sidebar.selectbox[0].options == ["coding", "qa", "summary"]
    at.switch_page("pages/01_ORCCA_Research_Audit.py").run()
    assert not at.exception
    assert at.title[0].value == "ORCCA Research Audit"
    assert len(at.tabs) == 5
    assert len(at.get("file_uploader")) == 0
    assert any("not run" in x.value for x in at.info)
    at.button(key="orcca_retrieve").click().run()
    assert not at.exception
    assert any("retrieved_evidence_unreviewed" in str(x.value) for x in at.markdown)
    assert at.button(key="orcca_generate").disabled
    assert len(at.get("download_button")) >= 2
    at.switch_page("app.py").run()
    assert not at.exception
    assert at.title[0].value == "ENVIO LLM Coding Assistant"


def test_local_private_upload_controls_require_both_operator_flags(monkeypatch):
    monkeypatch.setenv("ORCCA_DEPLOYMENT_MODE", "local_private")
    monkeypatch.setenv("ORCCA_ALLOW_PRIVATE_UPLOADS", "1")
    at = app().switch_page("pages/01_ORCCA_Research_Audit.py").run()
    assert not at.exception
    assert len(at.get("file_uploader")) == 8


def test_clear_rotates_upload_widgets_and_removes_retrieval(monkeypatch):
    monkeypatch.setenv("ORCCA_DEPLOYMENT_MODE", "local_private")
    monkeypatch.setenv("ORCCA_ALLOW_PRIVATE_UPLOADS", "1")
    at = app().switch_page("pages/01_ORCCA_Research_Audit.py").run()
    previous_keys = {w.key for w in at.get("file_uploader")}
    at.button(key="orcca_retrieve").click().run()
    assert "orcca_retrieval" in at.session_state
    at.button(key="orcca_clear").click().run()
    assert not at.exception
    assert not previous_keys.intersection(w.key for w in at.get("file_uploader"))
    assert "orcca_retrieval" not in at.session_state
    assert any("No private inputs supplied" in x.value for x in at.caption)


def test_private_upload_payloads_and_incomplete_inputs(monkeypatch):
    """Inject uploaded-file payloads; browser chooser is tested separately."""
    import io
    import json
    import streamlit
    from orcca_audit.demo import demo_data
    monkeypatch.setenv("ORCCA_DEPLOYMENT_MODE", "local_private")
    monkeypatch.setenv("ORCCA_ALLOW_PRIVATE_UPLOADS", "1")
    d = demo_data(); uploads = {}

    def file(name, data):
        f = io.BytesIO(data); f.name = name; return f

    uploads["orcca_upload_panel"] = file("synthetic_panel.csv", d["panel"].to_csv(index=False).encode())
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, key, **kwargs: uploads.get(key))
    at = app().switch_page("pages/01_ORCCA_Research_Audit.py").run()
    assert not at.exception
    assert any("will not be substituted" in x.value for x in at.error)
    assert not at.tabs
    for role in ["monthly", "annual", "tracker"]:
        uploads["orcca_upload_" + role] = file("synthetic_" + role + ".csv", d[role].to_csv(index=False).encode())
    uploads["orcca_upload_specs"] = file("synthetic.json", json.dumps(d["specifications"]).encode())
    uploads["orcca_upload_evidence"] = file("synthetic.txt", b"SYNTHETIC evidence. Unknown discontinuation needs review.")
    at.run()
    assert not at.exception
    assert not at.error, [x.value for x in at.error]
    assert len(at.tabs) == 5
    assert at.metric[0].value == "18"
    at.text_input(key="orcca_query").set_value("unknown discontinuation")
    at.button(key="orcca_retrieve").click().run()
    assert any("synthetic.txt" in str(x.value) for x in at.markdown)
    assert at.button(key="orcca_generate").disabled
    # A different evidence file must hide the earlier source result before
    # another retrieval, and numeric replacement must recalculate the summary.
    uploads["orcca_upload_evidence"] = file("replacement.txt", b"SYNTHETIC replacement. Zebra.")
    d["panel"].loc[0, ["Deaths", "y_lo", "y_hi"]] = 999
    uploads["orcca_upload_panel"] = file("changed.csv", d["panel"].to_csv(index=False).encode())
    at.run()
    assert at.metric[2].value == "17"
    assert not any("synthetic.txt" in str(x.value) for x in at.markdown)
    annual = uploads.pop("orcca_upload_annual")
    at.run()
    assert not at.tabs and not at.get("download_button")
    assert any("will not be substituted" in x.value for x in at.error)
    uploads["orcca_upload_annual"] = annual
    uploads["orcca_upload_specs"] = file("bad.json", b'{"a":1,"b":2}')
    at.run()
    assert not at.exception
    assert any("exactly two" in x.value for x in at.error)
    assert not at.tabs
