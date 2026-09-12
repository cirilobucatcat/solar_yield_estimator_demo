import datetime

from django import forms

# Placeholder list of PH cities/provinces. Swap this for whatever set of
# locations the Kaggle/weather data actually covers once Day 2 (feature
# engineering) is underway -- each entry should map to known average
# irradiation values used as model features.
PH_CITY_CHOICES = [
    ("manila", "Manila"),
    ("quezon_city", "Quezon City"),
    ("cebu_city", "Cebu City"),
    ("davao_city", "Davao City"),
    ("iloilo_city", "Iloilo City"),
    ("baguio_city", "Baguio City"),
    ("cagayan_de_oro", "Cagayan de Oro"),
    ("zamboanga_city", "Zamboanga City"),
    ("bacolod_city", "Bacolod City"),
    ("general_santos", "General Santos"),
]

# Defined here (not models.py) so both forms.py and models.py can use it
# without a circular import -- models.py already imports PH_CITY_CHOICES
# from this module.
MONTH_CHOICES = [
    (1, "January"), (2, "February"), (3, "March"), (4, "April"),
    (5, "May"), (6, "June"), (7, "July"), (8, "August"),
    (9, "September"), (10, "October"), (11, "November"), (12, "December"),
]


class SolarInputForm(forms.Form):
    system_size_kw = forms.DecimalField(
        label="System size (kW)",
        min_value=0.5,
        max_value=1000,
        max_digits=6,
        decimal_places=2,
        widget=forms.NumberInput(attrs={"step": "0.1", "class": "form-control"}),
    )
    panel_tilt_deg = forms.DecimalField(
        label="Panel tilt angle (°)",
        min_value=0,
        max_value=90,
        max_digits=4,
        decimal_places=1,
        initial=10.0,
        widget=forms.NumberInput(attrs={"step": "0.5", "class": "form-control"}),
    )
    city = forms.ChoiceField(
        label="City / Province",
        choices=PH_CITY_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    month = forms.TypedChoiceField(
        label="Month",
        choices=MONTH_CHOICES,
        coerce=int,
        initial=lambda: datetime.date.today().month,
        help_text="Solar output varies a lot by season in the PH -- defaults to this month.",
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    electricity_rate_override = forms.DecimalField(
        label="Electricity rate override (₱/kWh)",
        required=False,
        min_value=1,
        max_value=50,
        max_digits=6,
        decimal_places=2,
        help_text="Optional -- leave blank to use the default rate for your city/utility.",
        widget=forms.NumberInput(attrs={"step": "0.01", "class": "form-control",
                                          "placeholder": "e.g. 12.50"}),
    )
