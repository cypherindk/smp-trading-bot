"""
bist100_tickers.py
BIST 100 (XU100) hisse listesi — yfinance formatinda (.IS uzantili).

⚠️ ONEMLI: BIST 100 endeks bilesenleri yilda 4 kez (Ocak, Nisan, Temmuz,
Ekim donemlerinde) Borsa Istanbul tarafindan revize edilir. Asagidaki liste
uzun suredir endekste yer alan bilinen buyuk/orta olcekli sirketlerden
derlenmistir ama %100 guncel 100 bilesen garantisi vermez.

GUNCELLEME ONERISI (3 ayda bir):
  1. https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/xu100.aspx
     veya https://www.getmidas.com/canli-borsa/xu100-bist-100-hisseleri
     adresinden guncel XU100 bilesen listesine bakin.
  2. Asagidaki BIST100 sozlugune eksik/cikmis sembolleri ekleyin/silin.
  3. Yanlis/delisted bir sembol olsa bile kod hata vermez, o hisseyi
     "hata" diye loglayip diger hisselerle devam eder — riskli degildir,
     sadece o hisseyi taramamis olursunuz.

Sembol formati: yfinance icin TICKER + ".IS" (ornek: "THYAO.IS")
"""

BIST100 = [
    "AEFES", "AGHOL", "AKBNK", "AKSA", "AKSEN", "ALARK", "ALFAS", "ANSGR",
    "ARCLK", "ASELS", "ASTOR", "AVPGY", "BERA", "BIMAS", "BIOEN", "BRSAN",
    "BRYAT", "BUCIM", "CANTE", "CCOLA", "CIMSA", "CWENE", "DOAS", "DOHOL",
    "ECILC", "ECZYT", "EGEEN", "EKGYO", "ENJSA", "ENKAI", "EREGL", "EUPWR",
    "FROTO", "GARAN", "GESAN", "GLYHO", "GOLTS", "GUBRF", "GWIND", "HALKB",
    "HEKTS", "TRENJ", "ISCTR", "ISGYO", "ISMEN", "KARSN", "KAYSE", "KCAER",
    "KCHOL", "KLSER", "KMPUR", "KONTR", "KONYA", "KORDS", "TRMET", "TRALT",
    "KRDMD", "LOGO", "MAVI", "MGROS", "MIATK", "ODAS", "OTKAR", "OYAKC",
    "PENTA", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "SKBNK", "SMRTG",
    "SOKM", "TAVHL", "TCELL", "THYAO", "TKFEN", "TOASO", "TSKB", "TTKOM",
    "TTRAK", "TUKAS", "TUPRS", "TURSG", "ULKER", "VAKBN", "VESBE", "VESTL",
    "YEOTK", "YKBNK", "ZOREN", "DSTKF",
]

# yfinance icin ".IS" uzantili tam liste
BIST100_YF = [f"{t}.IS" for t in BIST100]