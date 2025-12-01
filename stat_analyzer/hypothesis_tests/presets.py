HYPOTHESES = [
    {
        "name": "genre_vs_global_sales",
        "cols": ["Genre", "Global_Sales"],
        "description": "Чи відрізняються глобальні продажі між різними жанрами ігор",
    },
    {
        "name": "platform_vs_global_sales",
        "cols": ["Platform", "Global_Sales"],
        "description": "Чи впливає платформа на середні глобальні продажі",
    },
    {
        "name": "na_vs_eu_correlation",
        "cols": ["NA_Sales", "EU_Sales"],
        "description": "Чи існує лінійний зв'язок між продажами в Північній Америці і Європі",
    },
    {
        "name": "genre_vs_platform",
        "cols": ["Genre", "Platform"],
        "description": "Чи пов'язані між собою жанр гри і платформа",
    },
]
