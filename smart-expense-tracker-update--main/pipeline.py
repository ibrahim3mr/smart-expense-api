import pandas as pd
import random

selected_features = [
    'avg_spend_ratio',
    'months_over_spending',
    'avg_food_pct_salary',
    'avg_drink_pct_salary',
    'avg_shopping_pct_salary',
    'avg_transport_pct_salary',
    'avg_bills_pct_salary',
    'avg_health_pct_salary',
    'avg_entertainment_pct_salary',
]

baseline_categories = [
    'avg_food_pct_salary',
    'avg_drink_pct_salary',
    'avg_shopping_pct_salary',
    'avg_transport_pct_salary',
    'avg_bills_pct_salary',
    'avg_health_pct_salary',
    'avg_entertainment_pct_salary'
]


def build_features(salary, spend_dict):

    salary = float(salary) if salary else 0.0
    salary_safe = salary if salary > 0 else 0.0

    total_spend = sum(float(v) for v in spend_dict.values())

    avg_spend_ratio = (total_spend / salary_safe) if salary_safe > 0 else 0.0

    months_over_spending = 1 if (salary_safe > 0 and total_spend > salary_safe) else 0

    data = {
        "avg_spend_ratio": avg_spend_ratio,
        "months_over_spending": months_over_spending,

        "avg_food_pct_salary": (spend_dict["food"] / salary_safe) if salary_safe > 0 else 0.0,
        "avg_drink_pct_salary": (spend_dict["drink"] / salary_safe) if salary_safe > 0 else 0.0,
        "avg_shopping_pct_salary": (spend_dict["shopping"] / salary_safe) if salary_safe > 0 else 0.0,
        "avg_transport_pct_salary": (spend_dict["transport"] / salary_safe) if salary_safe > 0 else 0.0,
        "avg_bills_pct_salary": (spend_dict["bills"] / salary_safe) if salary_safe > 0 else 0.0,
        "avg_health_pct_salary": (spend_dict["health"] / salary_safe) if salary_safe > 0 else 0.0,
        "avg_entertainment_pct_salary": (spend_dict["entertainment"] / salary_safe) if salary_safe > 0 else 0.0,
    }

    return pd.DataFrame([data])


def prepare_for_clustering(df):
    return df[selected_features]


def generate_final_recommendation(row, baseline_mean_df, threshold=0.05):

    categories_info = {

        'avg_food_pct_salary': {
            "name": "الأكل",
            "tips": {
                "low": [
                    "مصاريف الأكل أعلى شوية من الطبيعي — حاول تطبخ في البيت أكتر.",
                    "الأكل من برا ممكن يزود مصاريفك — جرب تعتمد على البيت.",
                    "تقليل المطاعم شوية ممكن يوفر جزء كويس من مرتبك.",
                    "حاول تخطط وجباتك الأسبوعية لتقليل مصاريف الأكل.",
                    "الأكل المنزلي غالباً أوفر وأصح.",
                    "لو قللت الأكل من برا هتلاحظ فرق في مصاريفك."
                ],
                "medium": [
                    "طلبات المطاعم بتأثر على ميزانيتك — حاول تقللها.",
                    "تخطيط الوجبات في البيت ممكن يقلل مصاريف الأكل.",
                    "لو قللت الأكل من برا شوية هتوفر مبلغ محترم.",
                    "جرب تحدد أيام معينة فقط للأكل من المطاعم.",
                    "الطبخ في البيت هيساعدك تتحكم في المصاريف.",
                    "تقليل الطلبات الخارجية هيكون له تأثير واضح."
                ],
                "high": [
                    "مصاريف الأكل عالية جداً — حاول تعتمد على الطبخ في البيت.",
                    "الأكل من برا بيأكل جزء كبير من مرتبك.",
                    "تقليل المطاعم هيكون له تأثير كبير على ميزانيتك.",
                    "الأفضل تحدد ميزانية شهرية للأكل من المطاعم.",
                    "الأكل الجاهز بيزود مصاريفك بشكل ملحوظ.",
                    "حاول تقلل الأكل من برا لأقل حد ممكن."
                ]
            }
        },

        'avg_drink_pct_salary': {
            "name": "المشروبات",
            "tips": {
                "low": [
                    "مصاريف المشروبات أعلى من الطبيعي شوية — حاول تقلل الكافيهات.",
                    "جرب تعمل القهوة في البيت بدل الكافيه.",
                    "تقليل زيارات الكافيه ممكن يوفر معاك فلوس.",
                    "المشروبات من الكافيهات بتزود المصاريف.",
                    "جرب تقلل القعدة في الكافيه شوية.",
                    "تحديد أيام للكافيه ممكن يساعدك توفر."
                ],
                "medium": [
                    "المشروبات بتاخد نسبة ملحوظة من مرتبك.",
                    "حدد عدد مرات للكافيه في الأسبوع.",
                    "لو عملت مشروباتك في البيت هتوفر مبلغ كويس.",
                    "تقليل المشروبات الجاهزة هيوفر مصاريف.",
                    "حاول تستبدل الكافيه بالمشروبات المنزلية.",
                    "تحديد ميزانية للمشروبات فكرة جيدة."
                ],
                "high": [
                    "مصاريف الكافيهات عالية جداً.",
                    "الكافيهات بتستهلك جزء كبير من مرتبك.",
                    "حدد ميزانية شهرية للمشروبات.",
                    "حاول تقلل الكافيهات بشكل واضح.",
                    "المشروبات الجاهزة بتأثر على ميزانيتك.",
                    "استبدال الكافيهات بالمشروبات المنزلية هيوفر كثير."
                ]
            }
        },

        'avg_shopping_pct_salary': {
            "name": "التسوق",
            "tips": {
                "low": [
                    "مصاريف التسوق أعلى شوية من الطبيعي.",
                    "فكر قبل أي شراء جديد.",
                    "حدد قائمة مشتريات قبل ما تشتري.",
                    "تجنب الشراء العشوائي.",
                    "اسأل نفسك هل تحتاج هذا الشيء فعلاً.",
                    "تحديد ميزانية للتسوق يساعدك تتحكم في المصاريف."
                ],
                "medium": [
                    "التسوق بياخد نسبة كبيرة من مرتبك.",
                    "اعمل قائمة بالمشتريات الضرورية.",
                    "تجنب الشراء العشوائي.",
                    "حدد سقف شهري للتسوق.",
                    "فكر مرتين قبل أي عملية شراء.",
                    "التخطيط للمشتريات يقلل المصاريف."
                ],
                "high": [
                    "مصاريف التسوق عالية جداً.",
                    "حدد ميزانية ثابتة للتسوق.",
                    "اشترِ الأشياء الضرورية فقط.",
                    "تجنب الشراء بدافع اللحظة.",
                    "حاول تقليل التسوق غير الضروري.",
                    "راجع مشترياتك الشهرية."
                ]
            }
        },

        'avg_transport_pct_salary': {
            "name": "المواصلات",
            "tips": {
                "low": [
                    "مصاريف المواصلات أعلى شوية من الطبيعي.",
                    "جرب تجمع مشاويرك في مرة واحدة.",
                    "خطط تنقلاتك لتقليل التكلفة.",
                    "استخدم وسائل نقل أوفر.",
                    "تقليل التنقل غير الضروري يوفر مصاريف.",
                    "تنظيم المشاوير يقلل استهلاك البنزين."
                ],
                "medium": [
                    "المواصلات بتاخد نسبة كبيرة من مرتبك.",
                    "استخدم المواصلات العامة أكتر.",
                    "جرب بدائل أوفر للتنقل.",
                    "شارك المشاوير إن أمكن.",
                    "تنظيم التنقلات يساعد في تقليل التكلفة.",
                    "فكر في وسائل نقل أقل تكلفة."
                ],
                "high": [
                    "مصاريف التنقل عالية جداً.",
                    "راجع مشاويرك وشوف إيه ممكن تقلله.",
                    "فكر في وسائل نقل أوفر.",
                    "تقليل التنقلات غير الضرورية مهم.",
                    "استخدام وسائل نقل عامة قد يوفر الكثير.",
                    "خطط تنقلاتك مسبقاً."
                ]
            }
        },

        'avg_bills_pct_salary': {
            "name": "الفواتير",
            "tips": {
                "low": [
                    "الفواتير أعلى شوية من الطبيعي.",
                    "راجع استهلاك الكهرباء والمياه.",
                    "تقليل الاستهلاك يوفر مبلغ كويس.",
                    "اقفل الأجهزة غير المستخدمة.",
                    "تابع استهلاكك الشهري.",
                    "تقليل الاستهلاك اليومي مهم."
                ],
                "medium": [
                    "الفواتير بتاخد نسبة ملحوظة من مرتبك.",
                    "تابع استهلاك الكهرباء والأجهزة.",
                    "راجع الاشتراكات الشهرية.",
                    "حاول تقليل الاستهلاك.",
                    "فكر في تقليل بعض الاشتراكات.",
                    "راجع استهلاكك بانتظام."
                ],
                "high": [
                    "الفواتير عالية جداً.",
                    "راجع كل الاشتراكات الشهرية.",
                    "شوف إيه ممكن تلغيه.",
                    "تقليل الاستهلاك ضروري.",
                    "تابع الفواتير الشهرية بدقة.",
                    "إلغاء الخدمات غير الضرورية مفيد."
                ]
            }
        },

        'avg_health_pct_salary': {
            "name": "الصحة",
            "tips": {
                "low": [
                    "مصاريف الصحة أعلى شوية من الطبيعي.",
                    "راجع المصاريف الطبية غير الضرورية.",
                    "لو عندك تأمين صحي استخدمه.",
                    "تأكد أن كل المصاريف ضرورية.",
                    "تابع مصاريف العلاج.",
                    "خطط للمصاريف الصحية."
                ],
                "medium": [
                    "المصاريف الصحية بتاخد نسبة كبيرة.",
                    "حاول تخطط المصاريف الطبية.",
                    "استفد من التأمين الصحي.",
                    "راجع تكاليف العلاج.",
                    "ابحث عن بدائل أوفر.",
                    "تنظيم المصاريف الصحية مهم."
                ],
                "high": [
                    "مصاريف الصحة عالية جداً.",
                    "حاول البحث عن بدائل أقل تكلفة.",
                    "خطط للمصاريف الصحية مسبقاً.",
                    "استخدم التأمين الصحي إن وجد.",
                    "راجع المصاريف الطبية.",
                    "التخطيط الصحي يقلل المصاريف."
                ]
            }
        },

        'avg_entertainment_pct_salary': {
            "name": "الترفيه",
            "tips": {
                "low": [
                    "مصاريف الترفيه أعلى شوية من الطبيعي.",
                    "حدد ميزانية للخروجات.",
                    "جرب أنشطة ترفيهية أقل تكلفة.",
                    "قلل الخروجات غير الضرورية.",
                    "اختار أنشطة مجانية أحياناً.",
                    "تنظيم الخروجات يقلل المصاريف."
                ],
                "medium": [
                    "الترفيه بياخد نسبة ملحوظة من مرتبك.",
                    "قلل عدد الخروجات الشهرية.",
                    "اختار أنشطة أوفر.",
                    "حدد ميزانية للترفيه.",
                    "جرب أنشطة أقل تكلفة.",
                    "تنظيم وقت الترفيه مهم."
                ],
                "high": [
                    "مصاريف الترفيه عالية جداً.",
                    "حدد سقف ثابت للترفيه.",
                    "قلل الخروجات لمرة أو مرتين في الشهر.",
                    "اختار أنشطة أقل تكلفة.",
                    "التقليل من الترفيه المدفوع يساعد.",
                    "تنظيم المصاريف الترفيهية مهم."
                ]
            }
        }
    }

    cluster_id   = row['cluster']
    spend_ratio  = float(row['avg_spend_ratio'])
    ratio_pct    = round(spend_ratio * 100)

    if spend_ratio < 0.5:
        level_msg = f"إنفاقك {ratio_pct}% من مرتبك — وضعك المالي ممتاز، كمّل كده"
    elif spend_ratio < 0.75:
        level_msg = f"إنفاقك {ratio_pct}% من مرتبك — وضعك مستقر وكويس"
    elif spend_ratio < 1.0:
        level_msg = f"إنفاقك {ratio_pct}% من مرتبك — قريب من الحد الأقصى، انتبه "
    else:
        overspend = round((spend_ratio - 1) * 100)
        level_msg = f"إنفاقك {ratio_pct}% من مرتبك — بتصرف أكتر من دخلك بنسبة {overspend}% "

    deviations = []

    for col, info in categories_info.items():
        current  = float(row[col])
        baseline = float(baseline_mean_df.loc[cluster_id, col])

        if baseline == 0:
            continue

        relative_diff = (current - baseline) / baseline

        if relative_diff > threshold:

            percent = round(relative_diff * 100)

            if percent < 30:
                severity = "low"
            elif percent < 60:
                severity = "medium"
            else:
                severity = "high"

            tip = random.choice(info["tips"][severity])

            deviations.append((info["name"], percent, tip))

    if not deviations:
        encouragements = [
            "مصروفاتك متوازنة في كل الفئات، استمر على نفس النهج!",
            "إنت بتتحكم في مصاريفك بشكل ممتاز.",
            "مصاريفك منظمة وفي حدودها الطبيعية."
        ]
        return f"{level_msg}\n\n {random.choice(encouragements)}"

    deviations_sorted = sorted(deviations, key=lambda x: x[1], reverse=True)

    tips_lines = []
    for name, percent, tip in deviations_sorted[:3]:
        tips_lines.append(f"• {tip} (أعلى من الطبيعي بـ {percent}%)")

    tips_text = "\n".join(tips_lines)

    return f"{level_msg}\n\n توصيات:\n{tips_text}"
