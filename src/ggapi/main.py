from fastapi import FastAPI, File, UploadFile
import uvicorn

from ggapi.asr import transcribe

app = FastAPI(title="GGAPi")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hello from ggapi!"}


@app.post("/transcribe")
def transcribe_audio(
    file: UploadFile = File(...), language: str | None = None
) -> dict[str, str]:
    return {"text": transcribe(file.file.read(), language)}


def main() -> None:
    uvicorn.run("ggapi.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
