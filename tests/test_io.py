import pandas as pd
from airline_revenue_analytics.io import replace_literal_N


def test_replace_literal_N_string_dtype():
    df = pd.DataFrame({"col": pd.Series(["\\N", "ok"], dtype="string")})
    out = replace_literal_N(df)
    assert pd.isna(out.loc[0, "col"])
    assert out.loc[1, "col"] == "ok"
