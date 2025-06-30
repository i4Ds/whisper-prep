import re


def remove_keywords_with_brackets(
    text,
    keywords=[
        "Live-Untertitel",
        "Wir untertiteln live",
        "Livepassagen",
        "1:1 Untertitelungen",
        "1:1-Untertitelung",
        "TELETEXT",
        "SWISS TXT",
        "Amara",
        "Untertitel",
        "ZDF",
        "Amara.org",
    ],
):
    if any(keyword in text for keyword in keywords):
        # Create a single pattern that matches any of the keywords
        pattern = re.compile(
            r"<\|.*?\|>.*?(" + "|".join(map(re.escape, keywords)) + r").*?<\|.*?\|>"
        )
        # Replace all matches with an empty string
        return pattern.sub("", text)
    else:
        return text


def normalize_abbrv(text):
    abbreviation_map = {
        "Std.": "Stunden",
        "Min.": "Minuten",
        "Sek.": "Sekunden",
        "Jr.": "Jahr",
        "Tg.": "Tage",
        "m ü.M.": "Meter über Meer",
        "u.a.": "unter anderem",
        "sog.": "sogenannte",
        "bzw.": "beziehungsweise",
        "ca.": "circa",
        "z.B.": "zum Beispiel",
        "d.h.": "das heisst",
        "D.h": "das heisst",
        "etc.": "et cetera",
        "z.T.": "zum Teil",
        "i.d.R.": "in der Regel",
        "u.v.m.": "und vieles mehr",
        "Nr.": "Nummer",
        "b.a.w.": "bis auf weiteres",
        "n.Chr.": "nach Christus",
        "v.Chr.": "vor Christus",
        "ggf.": "gegebenenfalls",
        "evtl.": "eventuell",
        "u.U.": "unter Umständen",
        "v.a.": "vor allem",
        "vgl.": "vergleiche",
        "inkl.": "inklusive",
        "zzgl.": "zuzüglich",
        "u.s.w.": "und so weiter",
        "bzgl.": "bezüglich",
        "s.a.": "siehe auch",
        "bspw.": "beispielsweise",
        "Hbf.": "Hauptbahnhof",
        "Dr.": "Doktor",
        "Dipl.": "Diplom",
        "Univ.": "Universität",
        "km/h": "Kilometer pro Stunde",
        "ms/s": "Meter pro Sekunde",
        "Gebr.": "Gebrüder",
        "äSEND$": " ",
        "Mio.": "Millionen",
        "Mia.": "Milliarden",
        "ähm": "",
    }
    for abbr, full in abbreviation_map.items():
        pattern = re.compile(rf"(^|\s)({re.escape(abbr)})(?=\s|$)")
        text = pattern.sub(lambda m: m.group(1) + full, text)
    return text


def standardize_text(text: str) -> str:
    """
    Clean, normalize, and remove specific symbols from text. This function:
    - Replaces certain Unicode characters and normalizes similar symbols to ASCII.
    - Removes specific unwanted symbols and quotation marks.
    - Collapses multiple spaces into a single space.
    """
    # Define replacements for specific Unicode and typographic characters
    replacements = {
        "\u00ad": "",  # Remove soft hyphen
        "ß": "ss",
        "/, ": "",  # Handle cases like 'Renter/, innen' mentioned in the comments
        "‘": "",
        "’": "",
        "“": "",
        "”": "",  # Remove different types of quotation marks
        "«": "",
        "»": "",
        "„": "",  # Remove additional types of quotation marks
        "(": "",
        ")": "",  # Remove parentheses
        "- -": "-",  # Replace double hyphens with a single hyphen
        "–": "-",
        "—": "-",  # Normalize en-dash and em-dash to hyphen
        '"': "",  # Remove double quotes
    }

    # Apply replacements
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    # Normalize multiple spaces to a single space, including handling cases where multiple spaces may form
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def normalize_capitalization(text):
    words = text.split()  # Split the text into words
    normalized_words = []

    for word in words:
        if len(word) > 4 and sum(1 for c in word if c.isupper()) == 2:
            # Make the entire word lowercase and then capitalize the first letter
            normalized_word = word[0].upper() + word[1:].lower()
        else:
            normalized_word = word
        normalized_words.append(normalized_word)

    return " ".join(normalized_words)


ELLIPSIS = r"(?:\.{3}|…)"              # ASCII or Unicode
def normalize_triple_dots(text: str) -> str:
    # ────────────────────────────────────────────────────────────────────────
    # 1️⃣  collapse any run of dots/ellipses to one period
    #     add a space _only_ if the next char begins a word/number
    # ────────────────────────────────────────────────────────────────────────
    def _collapse(match: re.Match) -> str:
        # peek at the next printable char (if any)
        after = match.string[match.end():match.end()+1]
        return ". " if after and after.isalnum() else "."

    text = re.sub(ELLIPSIS + r"\s*", _collapse, text)

    # ────────────────────────────────────────────────────────────────────────
    # 2️⃣  tidy punctuation spacing (same as before, but ellipses are gone)
    # ────────────────────────────────────────────────────────────────────────
    text = (
        text.replace(".,", ".")
            .replace("  ", " ")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" . ", ". ")
            .replace(". <", ".<")
            .replace("! <", "!<")
            .replace("? <", "?<")
            .replace(", <", ",<")
    )

    return text.strip()


def remove_bracketed_text(text):
    # Use regex to remove all text between [ and ] including the brackets
    return re.sub(r"\[.*?\]", "", text).strip()


def tokenize(text):
    # treat "..." as a standalone token
    return re.findall(r"\.\.\.|[^\s]+", text)


def collapse_ngrams(tokens, max_n=3):
    out, i = [], 0
    while i < len(tokens):

        # 1️⃣  Collapse runs of a single repeated token
        j = i + 1
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        if j - i > 1:  # found a run ≥ 2
            out.append(tokens[i])
            i = j
            continue  # restart loop

        # 2️⃣  Collapse repeated n-grams of length 2-max_n
        replaced = False
        for n in range(max_n, 1, -1):  # skip n=1 (already handled)
            if (
                i + 2 * n <= len(tokens)
                and tokens[i : i + n] == tokens[i + n : i + 2 * n]
            ):
                k = i + n
                while k + n <= len(tokens) and tokens[i : i + n] == tokens[k : k + n]:
                    k += n
                out.extend(tokens[i : i + n])  # keep one copy
                i = k
                replaced = True
                break
        if not replaced:
            out.append(tokens[i])
            i += 1
    return out


def collapse_text(text, max_n=3):
    return " ".join(collapse_ngrams(tokenize(text), max_n))


class GermanNumberConverter:
    def __init__(self):
        self.currency_map = {
            "$": "Dollar",
            "€": "Euro",
            "£": "Pfund",
            "¥": "Yen",
            "₹": "Rupie",
            "₽": "Rubel",
            "₩": "Won",
            "₫": "Dong",
            "₣": "Franken",
            "₦": "Naira",
            "₴": "Hrywnja",
            "₲": "Guarani",
            "₵": "Cedi",
            "฿": "Baht",
            "₺": "Türkische Lira",
            "₼": "Manat",
            "₪": "Schekel",
            "₭": "Kip",
            "₨": "Rupie",
            "₱": "Peso",
            "₳": "Austral",
            "₡": "Colon",
            "₢": "Cruzeiro",
            "₥": "Mill",
            "₧": "Peseta",
            "₮": "Tugrik",
            "₯": "Drachme",
            "₰": "Pfennig",
            "₷": "Speciedaler",
            "Fr.": "Franken",
            "CHF": "Franken",
            "USD": "Dollar",
            "EUR": "Euro",
            "GBP": "Pfund",
            "JPY": "Yen",
            "CNY": "Yuan",
            "INR": "Rupie",
            "RUB": "Rubel",
            "KRW": "Won",
            "VND": "Dong",
            "NGN": "Naira",
            "UAH": "Hrywnja",
            "GHS": "Cedi",
            "THB": "Baht",
            "TRY": "Türkische Lira",
            "AZN": "Manat",
            "ILS": "Schekel",
            "LAK": "Kip",
            "PKR": "Rupie",
            "PHP": "Peso",
            "ARS": "Peso",
            "CRC": "Colon",
            "BRL": "Real",
            "CLP": "Peso",
            "MXN": "Peso",
            "SGD": "Dollar",
            "HKD": "Dollar",
        }

    def combine_numbers(self, text):
        # This regex looks for numbers separated by a space where the second number is entirely zero, e.g. 150 000
        return re.sub(r"(\d+)\s+(0+)(?!\d)", r"\1\2", text)

    def remove_apostrophes(self, text):
        # Use regex to remove apostrophes surrounded by digits
        return re.sub(r"(?<=\d)'(?=\d)", "", text)

    def replace_komma_w_dot(self, text):
        # Normalizes a comma to a dot for numbers
        # For example, "1,234" becomes "1.234"
        return re.sub(r"(?<=\d),(?=\d)", ".", text)

    def replace_currencies(self, text):
        for symbol, word in self.currency_map.items():
            text = text.replace(symbol, word)
        return text

    def convert_number(self, text):
        text = self.replace_currencies(text)
        text = self.combine_numbers(text)
        text = self.remove_apostrophes(text)
        text = self.replace_komma_w_dot(text)
        return text

    def convert(self, text):
        return self.convert_number(text)


def normalize_text(text):
    text = remove_bracketed_text(text)
    text = normalize_triple_dots(text)
    text = normalize_abbrv(text)
    text = normalize_capitalization(text)
    text = standardize_text(text)
    text = collapse_text(text)
    converter = GermanNumberConverter()
    text = converter.convert(text)
    return text
