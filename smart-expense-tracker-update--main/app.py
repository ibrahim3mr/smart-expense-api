import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

from pipeline import build_features, generate_final_recommendation, prepare_for_clustering


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Smart Expense Tracker",
    layout="centered"
)

# -------------------------------
# Title
# -------------------------------
st.title("💰 Smart Expense Tracker")
st.write("أدخل مصاريف الشهر كله عشان نقدر نحلل وضعك المالي ونديك توصيات دقيقة.")

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    scaler          = joblib.load("robust_scaler.joblib")
    kmeans          = joblib.load("kmeans_model.joblib")
    cluster_baseline = joblib.load("cluster_baseline.joblib")
    return scaler, kmeans, cluster_baseline


scaler, kmeans, cluster_baseline = load_models()

# -------------------------------
# User Inputs
#  FIX 1: وضحنا إن الأرقام دي مصاريف الشهر كله
# -------------------------------
st.subheader(" بيانات الشهر")

salary = st.number_input("المرتب الشهري *", min_value=0.0, step=100.0)

st.markdown("---")
st.markdown("**أدخل مصاريف الشهر كله (مش اليوم)**")

col1, col2 = st.columns(2)

with col1:
    food          = st.number_input(" الأكل (الشهر كله)",          min_value=0.0, step=10.0)
    drink         = st.number_input(" المشروبات (الشهر كله)",       min_value=0.0, step=10.0)
    shopping      = st.number_input(" التسوق (الشهر كله)",         min_value=0.0, step=10.0)
    transport     = st.number_input(" المواصلات (الشهر كله)",       min_value=0.0, step=10.0)

with col2:
    bills         = st.number_input(" الفواتير (الشهر كله)",        min_value=0.0, step=10.0)
    health        = st.number_input(" الصحة (الشهر كله)",           min_value=0.0, step=10.0)
    entertainment = st.number_input(" الترفيه (الشهر كله)",         min_value=0.0, step=10.0)

# -------------------------------
# Button
# -------------------------------
if st.button("🔍 تحليل مصاريفي"):

    if salary <= 0:
        st.error(" لازم تدخل المرتب الشهري الأول علشان نقدر نحلل مصاريفك.")
        st.stop()

    # -------------------------------
    # Expenses Dictionary
    # -------------------------------
    spend_dict = {
        "food":          float(food),
        "drink":         float(drink),
        "shopping":      float(shopping),
        "transport":     float(transport),
        "bills":         float(bills),
        "health":        float(health),
        "entertainment": float(entertainment)
    }

    total_spend = sum(spend_dict.values())

    # -------------------------------
    # Summary Card
    # -------------------------------
    st.markdown("---")
    st.subheader(" ملخص الشهر")

    m1, m2, m3 = st.columns(3)
    m1.metric("إجمالي المصاريف", f"{total_spend:,.0f}")
    m2.metric("المرتب", f"{salary:,.0f}")
    remaining = salary - total_spend
    m3.metric("المتبقي", f"{remaining:,.0f}", delta=f"{remaining:,.0f}")

    if total_spend > salary:
        st.warning("⚠️ مصروفاتك أكبر من مرتبك الشهر ده — حاول تراجع أكبر فئة وتقللها.")

    # -------------------------------
    # Feature Engineering & Clustering
    # -------------------------------
    new_user_df = build_features(salary, spend_dict)
    X_new       = prepare_for_clustering(new_user_df)
    X_scaled    = scaler.transform(X_new)
    cluster     = kmeans.predict(X_scaled)[0]
    new_user_df['cluster'] = cluster

    # -------------------------------
    # Recommendation
    #  FIX 3 + 4: التوصية دلوقتي بتاخد severity + سطر لكل category
    # -------------------------------
    st.markdown("---")
    st.subheader(" تحليلك المالي وتوصياتك")

    recommendation = generate_final_recommendation(
        new_user_df.iloc[0],
        cluster_baseline
    )

    st.info(recommendation)

    # -------------------------------
    # Chart
    # -------------------------------
    st.markdown("---")
    st.subheader(" توزيع المصروفات")

    chart_data = pd.DataFrame({
        "الفئة": [
            "الأكل", "المشروبات", "التسوق",
            "المواصلات", "الفواتير", "الصحة", "الترفيه"
        ],
        "المبلغ": [
            spend_dict["food"],    spend_dict["drink"],   spend_dict["shopping"],
            spend_dict["transport"], spend_dict["bills"], spend_dict["health"],
            spend_dict["entertainment"]
        ]
    })

    # نسبة كل فئة
    chart_data["النسبة %"] = (chart_data["المبلغ"] / salary * 100).round(1)

    fig = px.bar(
        chart_data,
        x="الفئة",
        y="المبلغ",
        text="النسبة %",
        color="الفئة",
        hover_data=["المبلغ", "النسبة %"]
    )

    fig.update_traces(
        texttemplate="%{text}%",
        textposition="outside"
    )

    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14),
        xaxis_title="",
        yaxis_title="المبلغ (جنيه)",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Pie Chart — نسب الفئات
    # -------------------------------
    fig_pie = px.pie(
        chart_data,
        names="الفئة",
        values="المبلغ",
        title="نسبة كل فئة من إجمالي المصاريف"
    )

    fig_pie.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=13)
    )

    st.plotly_chart(fig_pie, use_container_width=True)
