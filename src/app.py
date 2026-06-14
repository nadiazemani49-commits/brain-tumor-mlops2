import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="Diagnostic IRM Cérébrale — PFA 2024/2025",
    page_icon="🧠",
    layout="wide"
)

with st.sidebar:
    st.markdown("### 🧠 Brain Tumor AI")
    st.markdown("---")
    st.markdown("**PFA 2024/2025**")
    st.markdown("👩‍🎓 Nadia Zemani")
    st.markdown("👨‍🏫 Pr. Mohamed LAZAAR")
    st.markdown("---")
    st.markdown("**Modèle : EfficientNet-B3**")
    st.markdown("**val_acc : 94.37%**")
    st.markdown("**F1-macro : 0.944**")
    st.markdown("---")
    try:
        r = requests.get("http://localhost:8000/health", timeout=2)
        st.success("✅ API connectée") if r.status_code == 200 else st.error("❌ API non disponible")
    except:
        st.error("❌ API non disponible")
    try:
        r_model = requests.get("http://localhost:8000/model/info", timeout=3)
        if r_model.status_code == 200:
            m = r_model.json()
            st.markdown("---")
            st.success(f"🏆 v{m['version']} — {m['stage']}")
            st.caption(f"val_acc: 94.37% | F1: 0.944")
    except:
        pass

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1B4F8A, #0D9488);
    padding: 2rem; border-radius: 12px;
    color: white; text-align: center; margin-bottom: 2rem;
}
</style>
<div class="main-header">
    <h1>🧠 Plateforme Intelligente de Diagnostic IRM Cérébrale</h1>
    <p>EfficientNet-B3 • RAG PubMed • MLOps Pipeline</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Diagnostic", "📊 Statistiques", "📋 Historique", "🔍 Monitoring"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DIAGNOSTIC
# ══════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Upload de l'image IRM")
        uploaded_file = st.file_uploader(
            "Glisse une image IRM (.jpg, .png)",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image IRM uploadée", use_column_width=True)
            st.info(f"Taille : {image.size[0]}×{image.size[1]} pixels")

            if st.button("🔍 Analyser l'IRM", type="primary", use_container_width=True):
                with st.spinner("⏳ Analyse en cours..."):
                    try:
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                        response = requests.post("http://localhost:8000/predict", files=files, timeout=30)
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['last_result'] = result
                            if 'history' not in st.session_state:
                                st.session_state['history'] = []
                            st.session_state['history'].append({
                                'timestamp' : datetime.now().strftime('%H:%M:%S'),
                                'filename'  : uploaded_file.name,
                                'prediction': result['prediction'],
                                'confidence': result['confidence'],
                                'urgent'    : result['urgent'],
                            })
                            st.success("✅ Analyse terminée !")
                        else:
                            st.error(f"Erreur API : {response.status_code}")
                    except Exception as e:
                        st.error(f"Erreur : {str(e)}")

    with col2:
        st.markdown("### 📋 Résultat du diagnostic")
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            colors_map = {
                'ÉLEVÉE' : ('#FEE2E2', '#DC2626'),
                'MODÉRÉE': ('#FEF9C3', '#CA8A04'),
                'FAIBLE' : ('#DCFCE7', '#16A34A'),
            }
            bg, fg = colors_map.get(result['severity'], ('#F3F4F6', '#374151'))
            urgency = "🚨 URGENT" if result['urgent'] else "✅ NON URGENT"
            st.markdown(f"""
            <div style="background:{bg};border:2px solid {fg};
                        padding:1.5rem;border-radius:8px;margin-bottom:1rem">
                <h2 style="color:{fg};margin:0">{urgency}</h2>
                <h3 style="color:{fg}">Diagnostic : {result['prediction_fr']}</h3>
                <p style="color:{fg}">Confiance : <b>{result['confidence']*100:.1f}%</b></p>
                <p style="color:{fg}">Sévérité : <b>{result['severity']}</b></p>
            </div>
            """, unsafe_allow_html=True)

            probs = result['probabilities']
            classes_fr = {'glioma':'Gliome','meningioma':'Méningiome',
                         'notumor':'Normal','pituitary':'Hypophysaire'}
            fig = go.Figure(go.Bar(
                x=list(probs.values()),
                y=[classes_fr[c] for c in probs.keys()],
                orientation='h',
                marker_color=['#E53E3E','#D69E2E','#38A169','#3182CE'],
                text=[f"{v*100:.1f}%" for v in probs.values()],
                textposition='outside',
            ))
            fig.update_layout(title="Probabilités par classe",
                             xaxis=dict(range=[0,1.15]), height=280,
                             margin=dict(l=10,r=10,t=40,b=10), plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

            col_a, col_b, col_c = st.columns(3)
            with col_a: st.metric("Confiance", f"{result['confidence']*100:.1f}%")
            with col_b: st.metric("Incertitude", f"{result['uncertainty']:.3f}")
            with col_c: st.metric("Modèle", "EfficientNet-B3")
        else:
            st.info("👆 Upload une image IRM et clique sur Analyser")

# ══════════════════════════════════════════════════════════════
# TAB 2 — STATISTIQUES
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Performance du modèle EfficientNet-B3")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("val_acc", "94.37%", "+15.67%")
    with col2: st.metric("F1-macro", "0.944", "+0.158")
    with col3: st.metric("AUC-ROC", "0.972", "Excellent")
    with col4: st.metric("Best epoch", "34/40")

    col1, col2 = st.columns(2)
    with col1:
        df = pd.DataFrame({
            'Classe'  : ['Gliome','Méningiome','Hypophysaire','Normal'],
            'Accuracy': [69.05, 87.56, 95.45, 99.23],
            'AUC-ROC' : [93.40, 96.50, 99.33, 99.63],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accuracy (%)', x=df['Classe'], y=df['Accuracy'],
                            marker_color=['#E53E3E','#D69E2E','#3182CE','#38A169']))
        fig.add_trace(go.Bar(name='AUC-ROC (%)', x=df['Classe'], y=df['AUC-ROC'],
                            marker_color=['#FC8181','#F6AD55','#63B3ED','#68D391']))
        fig.update_layout(title="Métriques par classe", barmode='group',
                         yaxis=dict(range=[0,110]), height=350, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=[78.7, 78.6, 85.0, 100.0, 75.0],
            theta=['val_acc','F1-macro','AUC-ROC','notumor','glioma'],
            fill='toself', name='10% dataset', line_color='#FC8181'))
        fig2.add_trace(go.Scatterpolar(
            r=[94.37, 94.4, 97.2, 99.2, 69.1],
            theta=['val_acc','F1-macro','AUC-ROC','notumor','glioma'],
            fill='toself', name='100% dataset', line_color='#38A169'))
        fig2.update_layout(title="10% vs 100% dataset",
                          polar=dict(radialaxis=dict(range=[0,100])), height=350)
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — HISTORIQUE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 Historique des diagnostics")
    if 'history' in st.session_state and st.session_state['history']:
        df_hist = pd.DataFrame(st.session_state['history'])
        df_hist['confidence'] = df_hist['confidence'].apply(lambda x: f"{x*100:.1f}%")
        df_hist['urgent'] = df_hist['urgent'].apply(lambda x: "🚨 Urgent" if x else "✅ Normal")
        df_hist.columns = ['Heure','Fichier','Diagnostic','Confiance','Urgence']
        st.dataframe(df_hist, use_container_width=True)
        if st.button("🗑️ Effacer l'historique"):
            st.session_state['history'] = []
            st.rerun()
    else:
        st.info("Aucun diagnostic effectué pour le moment.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — MONITORING
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🔍 Monitoring & Observabilité en Production")

    try:
        r = requests.get("http://localhost:8000/monitoring", timeout=3)
        stats = r.json()

        if stats.get('drift_detected'):
            st.error("🚨 ALERTE : Data drift détecté ! Confiance moyenne faible.")
        else:
            st.success("✅ Système stable — aucun drift détecté")

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total prédictions", stats.get('total_predictions', 0))
        with col2:
            conf = stats.get('avg_confidence', 0)
            st.metric("Confiance moyenne", f"{conf*100:.1f}%",
                     "⚠️ Faible" if conf < 0.7 else "✅ OK")
        with col3: st.metric("Incertitude moyenne", f"{stats.get('avg_uncertainty', 0):.3f}")
        with col4: st.metric("Dernière prédiction", str(stats.get('last_prediction', 'N/A'))[:19])

        class_dist = stats.get('class_distribution', {})
        if class_dist:
            st.markdown("#### Distribution des classes en production")
            classes_fr = {'glioma':'Gliome','meningioma':'Méningiome',
                         'notumor':'Normal','pituitary':'Hypophysaire'}

            col1, col2 = st.columns(2)
            with col1:
                labels = [classes_fr.get(k, k) for k in class_dist.keys()]
                values = list(class_dist.values())
                fig = go.Figure(go.Pie(
                    labels=labels, values=values,
                    marker_colors=['#E53E3E','#D69E2E','#38A169','#3182CE'],
                    hole=0.4
                ))
                fig.update_layout(title="Répartition des diagnostics", height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                train_dist = {'Gliome':23.1,'Méningiome':23.4,'Hypophysaire':25.0,'Normal':28.5}
                total = sum(values) or 1
                prod_dist = {classes_fr.get(k,k): v/total*100 for k,v in class_dist.items()}
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(name='Train (référence)',
                    x=list(train_dist.keys()), y=list(train_dist.values()),
                    marker_color='#4299E1', opacity=0.7))
                fig2.add_trace(go.Bar(name='Production (actuel)',
                    x=list(prod_dist.keys()), y=list(prod_dist.values()),
                    marker_color='#F6AD55', opacity=0.7))
                fig2.update_layout(barmode='group', title="Train vs Production",
                                  yaxis_title="%", height=350, plot_bgcolor='white')
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Rapport de Data Drift Evidently AI")
        if st.button("📊 Générer rapport Evidently AI", type="primary"):
            with st.spinner("Génération..."):
                r2 = requests.get("http://localhost:8000/monitoring/report", timeout=30)
                result = r2.json()
                if result.get('status') == 'ok':
                    st.success("✅ Rapport généré !")
                    st.code("\\\\wsl$\\Ubuntu\\home\\nadia\\brain-tumor-mlops\\logs\\drift_report.html")
                    st.info("Copie ce chemin dans Windows Explorer pour ouvrir le rapport")
                else:
                    st.warning(result.get('message', 'Pas assez de données — fais 50+ prédictions'))

    except Exception as e:
        st.warning(f"API non disponible : {str(e)}")
        st.info("Lance l'API : sudo docker compose up -d")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:gray;font-size:12px'>
⚠️ Outil d'aide au diagnostic — doit être validé par un médecin<br>
PFA 2024/2025 | Nadia Zemani | Pr. Mohamed LAZAAR | ENSIAS Rabat
</div>
""", unsafe_allow_html=True)
