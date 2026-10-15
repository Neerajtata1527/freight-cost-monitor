from pathlib import Path

import pandas as pd
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTEXT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "context_notes.csv"
)

REQUIRED_COLUMNS = [
    "note_id",
    "date",
    "applies_to",
    "note",
]


def load_context_notes() -> list[Document]:
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            f"Context notes file not found: {CONTEXT_FILE}"
        )

    df = pd.read_csv(CONTEXT_FILE)

    missing_columns = (
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    for column in ["note_id", "applies_to", "note"]:
        df[column] = df[column].astype("string").str.strip()

    if df.isnull().any().any():
        raise ValueError(
            "Context notes contain missing values."
        )

    if df.duplicated(subset=["note_id"]).any():
        raise ValueError("Duplicate note IDs found.")

    for column in ["note_id", "applies_to", "note"]:
        if df[column].eq("").any():
            raise ValueError(
                f"Empty values found in {column}."
            )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")

    documents = []

    for _, row in df.iterrows():
        note_id = str(row["note_id"])
        note_date = str(row["date"])
        applies_to = str(row["applies_to"])
        note_text = str(row["note"])

        document = Document(
            page_content=(
                f"Context note: {note_text}\n"
                f"Applicable route: {applies_to}\n"
                f"Effective date: {note_date}"
            ),
            metadata={
                "note_id": note_id,
                "date": note_date,
                "applies_to": applies_to,
                "applies_to_normalized": applies_to.casefold(),
                "note": note_text,
            },
        )

        documents.append(document)

    return documents


if __name__ == "__main__":
    notes = load_context_notes()

    print(f"Loaded {len(notes)} context notes.")

    for document in notes[:3]:
        print(document.metadata)
        print(document.page_content)
        print("-" * 50)