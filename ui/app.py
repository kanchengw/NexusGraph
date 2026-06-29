
import uuid, requests, streamlit as st
st.set_page_config(page_title="NexusGraph", page_icon="\U0001f52e", layout="centered")
API = "http://localhost:8000/api/v1"

if "msgs" not in st.session_state: st.session_state.msgs = []
if "sid" not in st.session_state: st.session_state.sid = str(uuid.uuid4())[:8]
if "rated" not in st.session_state: st.session_state.rated = set()

st.title("\U0001f52e NexusGraph")

for i, m in enumerate(st.session_state.msgs):
    with st.chat_message(m["role"]):
        st.write(m["txt"])
        if m["role"] == "assistant" and m.get("src"):
            with st.expander("Sources"):
                for s in m["src"]: st.caption(s)
    if m["role"] == "assistant" and i not in st.session_state.rated:
        c = st.columns([5, 1, 1, 1, 1, 1])
        c[0].caption("Rate this answer:")
        for ri in range(5):
            if c[ri+1].button(str(ri+1), key=f"rb_{i}_{ri}"):
                q = st.session_state.msgs[i-1]["txt"] if i > 0 else ""
                a = m["txt"]
                ok = False
                try:
                    rr = requests.post(f"{API}/graphrag/feedback", json={
                        "session_id": st.session_state.sid, "query": q,
                        "response": a, "rating": ri+1
                    }, timeout=5)
                    ok = rr.ok
                except: pass
                if ok:
                    st.session_state.rated.add(i)
                    st.rerun()
    elif m["role"] == "assistant" and i in st.session_state.rated:
        st.caption("\u2705 Thank you for your feedback")

p = st.chat_input("Ask a question...")
if p:
    st.session_state.msgs.append({"role": "user", "txt": p})
    with st.chat_message("user"): st.write(p)
    with st.chat_message("assistant"):
        ph = st.empty()
        with st.spinner(""):
            try:
                r = requests.post(f"{API}/graphrag/query", json={"question": p, "top_k": 5}, timeout=90)
                d = r.json() if r.ok else None
            except: d = None
        if d:
            a = d.get("answer", "") or "(empty)"
            ph.write(a)
            src = d.get("source_docs", [])
            if src:
                with st.expander("Sources"):
                    for s in src: st.caption(s)
            st.session_state.msgs.append({"role": "assistant", "txt": a, "src": src})
            st.rerun()
        else:
            ph.error("Server unreachable")
