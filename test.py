from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
import base64
import json
import requests
import re
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from io import BytesIO
from uuid import uuid4
from datetime import datetime, timezone
from collections import deque
from pathlib import Path

# 🔑 OpenRouter API Key from environment
API_KEY = "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"

# 🌐 OpenRouter API Endpoint
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# ✅ Model
MODEL = "google/gemini-2.5-flash-image-preview"

# Simple file-based recent store
ZSTORE_DIR = Path("e:/Internship/cal_ai/.zstore")
ZSTORE_FILE = ZSTORE_DIR / "meals.json"

app = FastAPI(
    title="🍽️ AI Meal Nutrition Analyzer",
    description="Analyze meal images and estimate nutrition using AI.",
    version="1.1"
)

# Add root route to fix 404 error
@app.get("/")
async def root():
    return {"message": "Welcome to the AI Meal Nutrition Analyzer API!"}

# --- Pydantic Models ---

class FoodItem(BaseModel):
    name: str
    weight: float

class ItemOut(BaseModel):
    name: str
    weight: float
    calories: float
    protein: float
    carbs: float
    fats: float
    ingredients: List[str] = Field(default_factory=list)

class Totals(BaseModel):
    calories: float
    protein: float
    carbs: float
    fats: float

class AnalysisResponse(BaseModel):
    id: str
    timestamp: str
    image_meta: Dict[str, Any]
    meal_name: str
    items: List[ItemOut]
    total: Totals
    confidence_score: int

class UpdateRequest(BaseModel):
    new_items: List[FoodItem]

class AddFoodRequest(BaseModel):
    current_meal: AnalysisResponse
    new_items: List[FoodItem]

# --- Helper Functions ---

def encode_image_from_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')


def _ensure_api_key():
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENROUTER_API_KEY environment variable")


def call_vision(prompt: str, image_b64: str, mime_type: str = "image/jpeg") -> dict:
    _ensure_api_key()
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data_url = f"data:{mime_type};base64,{image_b64}"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        "temperature": 0.2
    }

    try:
        res = requests.post(BASE_URL, headers=headers, json=payload)
        if res.status_code != 200:
            raise Exception(f"API Error {res.status_code}: {res.text}")

        content = res.json()["choices"][0]["message"]["content"].strip()
        # Extract JSON from ```json block when present
        match = re.search(r"```(?:json)?\s*({.*?})\s*```", content, re.DOTALL)
        json_str = match.group(1) if match else content
        return json.loads(json_str)

    except Exception as e:
        print(f"call_vision failed: {e}")
        raise


def estimate_nutrition(name: str, weight: float) -> dict:
    """Estimate nutrition for a food item via LLM; fall back to zeros on error."""
    try:
        _ensure_api_key()
    except HTTPException:
        return {"calories": 0, "protein": 0, "carbs": 0, "fats": 0}

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = (
        f"Estimate nutrition for {weight}g of {name}. Respond exactly in JSON: "
        f"{{\"calories\": number, \"protein\": number, \"carbs\": number, \"fats\": number}}"
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }

    try:
        res = requests.post(BASE_URL, headers=headers, json=payload)
        if res.status_code != 200:
            return {"calories": 0, "protein": 0, "carbs": 0, "fats": 0}

        content = res.json()["choices"][0]["message"]["content"].strip()
        json_match = re.search(r'({.*})', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        return json.loads(content)

    except Exception:
        return {"calories": 0, "protein": 0, "carbs": 0, "fats": 0}


def coerce_number(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def normalize_items(items: List[Dict[str, Any]]) -> List[ItemOut]:
    norm: List[ItemOut] = []
    for it in items or []:
        ingredients = it.get("ingredients")
        if not isinstance(ingredients, list):
            ingredients = []
        norm.append(
            ItemOut(
                name=str(it.get("name", "item")).strip(),
                weight=coerce_number(it.get("weight")),
                calories=coerce_number(it.get("calories")),
                protein=coerce_number(it.get("protein")),
                carbs=coerce_number(it.get("carbs")),
                fats=coerce_number(it.get("fats")),
                ingredients=[str(x) for x in ingredients],
            )
        )
    return norm


def sum_totals(items: List[ItemOut]) -> Totals:
    cals = sum(i.calories for i in items)
    prot = sum(i.protein for i in items)
    carbs = sum(i.carbs for i in items)
    fats = sum(i.fats for i in items)
    return Totals(calories=cals, protein=prot, carbs=carbs, fats=fats)


def update_with_new_item(meal_data: AnalysisResponse, name: str, weight: float) -> AnalysisResponse:
    nut = estimate_nutrition(name, weight)
    new_item = ItemOut(
        name=name,
        weight=weight,
        calories=coerce_number(nut.get("calories", 0)),
        protein=coerce_number(nut.get("protein", 0)),
        carbs=coerce_number(nut.get("carbs", 0)),
        fats=coerce_number(nut.get("fats", 0)),
        ingredients=[],
    )
    items = list(meal_data.items) + [new_item]
    totals = sum_totals(items)
    updated = AnalysisResponse(
        id=meal_data.id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        image_meta=meal_data.image_meta,
        items=items,
        total=totals,
        confidence_score=meal_data.confidence_score,
    )
    return updated


# Recent meals persistence (very lightweight)
_recent_cache: deque = deque(maxlen=50)

def _zstore_init():
    try:
        ZSTORE_DIR.mkdir(parents=True, exist_ok=True)
        if not ZSTORE_FILE.exists():
            ZSTORE_FILE.write_text(json.dumps({"meals": []}, ensure_ascii=False))
    except Exception as e:
        print(f"recent store init failed: {e}")


def save_recent(meal: AnalysisResponse):
    # memory cache
    _recent_cache.appendleft(meal.dict())
    # file store best-effort
    try:
        _zstore_init()
        data = json.loads(ZSTORE_FILE.read_text() or '{}')
        meals = data.get("meals", [])
        meals.insert(0, meal.dict())
        meals = meals[:50]
        ZSTORE_FILE.write_text(json.dumps({"meals": meals}, ensure_ascii=False))
    except Exception as e:
        print(f"save_recent failed: {e}")


def get_recent(limit: int = 10) -> List[dict]:
    out: List[dict] = []
    try:
        # memory first
        out.extend(list(_recent_cache)[:limit])
        if len(out) >= limit:
            return out[:limit]
        # then file
        if ZSTORE_FILE.exists():
            data = json.loads(ZSTORE_FILE.read_text() or '{}')
            meals = data.get("meals", [])
            out.extend(meals[: max(0, limit - len(out))])
    except Exception as e:
        print(f"get_recent failed: {e}")
    return out[:limit]


# --- FastAPI Endpoints ---

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_meal_endpoint(image: UploadFile = File(...), new_items: Optional[str] = Form(None)):
    """
    Analyze a meal image and return nutritional breakdown with ingredients.
    Optionally add custom food items.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Read image bytes
    try:
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file uploaded")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    # Encode to base64
    try:
        image_b64 = encode_image_from_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base64 encoding failed: {str(e)}")

    # Detect MIME type
    ext = (image.filename or "").split(".")[-1].lower()
    mime_type = "image/png" if ext == "png" else "image/jpeg"

    # Vision prompt requiring ingredients
    prompt = (
        "Analyze this meal image. Give the meal a short, descriptive name. "
        "For each visible food item, estimate strictly: name, weight (grams), calories, protein (g), carbs (g), fats (g), and ingredients (array of strings). "
        "Then compute overall totals and a confidence_score 0-100. Respond exactly in JSON: \n"
        "{\"meal_name\": \"...\",\n"
        "  \"items\": [\n"
        "    {\"name\": \"...\", \"weight\": num, \"calories\": num, \"protein\": num, \"carbs\": num, \"fats\": num, \"ingredients\": [\"...\"]}\"\n"
        "  ],\n"
        "  \"total\": {\"calories\": num, \"protein\": num, \"carbs\": num, \"fats\": num},\n"
        "  \"confidence_score\": integer\n"
        "}\n"
        "No extra text."
    )

    # Call vision

    try:
        raw_result = call_vision(prompt, image_b64, mime_type)
    except Exception:
        # Fallback minimal response
        fallback = AnalysisResponse(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            image_meta={"filename": image.filename, "mime": mime_type},
            meal_name="Unknown Meal",
            items=[],
            total=Totals(calories=0, protein=0, carbs=0, fats=0),
            confidence_score=0,
        )
        save_recent(fallback)
        return fallback

    if not isinstance(raw_result, dict):
        raise HTTPException(status_code=500, detail="Invalid response format from AI")

    meal_name = raw_result.get("meal_name", "Unnamed Meal")
    items_norm = normalize_items(raw_result.get("items", []))

    # Optionally add new items
    if new_items:
        try:
            items_data = json.loads(new_items)
            for item in items_data:
                name = item.get("name")
                weight = item.get("weight")
                if name and isinstance(weight, (int, float)) and weight > 0:
                    nut = estimate_nutrition(name, float(weight))
                    items_norm.append(
                        ItemOut(
                            name=name,
                            weight=float(weight),
                            calories=coerce_number(nut.get("calories", 0)),
                            protein=coerce_number(nut.get("protein", 0)),
                            carbs=coerce_number(nut.get("carbs", 0)),
                            fats=coerce_number(nut.get("fats", 0)),
                            ingredients=[],
                        )
                    )
        except json.JSONDecodeError:
            pass

    totals = sum_totals(items_norm)
    confidence = int(coerce_number(raw_result.get("confidence_score", 0)))
    resp = AnalysisResponse(
        id=str(uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        image_meta={"filename": image.filename, "mime": mime_type},
        meal_name=meal_name,
        items=items_norm,
        total=totals,
        confidence_score=max(0, min(100, confidence)),
    )

    save_recent(resp)
    return resp


@app.post("/add-food", response_model=AnalysisResponse)
async def add_food_item(req: AddFoodRequest):
    """
    Add one or more food items to an existing meal analysis and return updated totals.
    """
    meal = req.current_meal
    # Ensure items are ItemOut objects
    items = [ItemOut(**i.dict()) if isinstance(i, BaseModel) else ItemOut(**i) for i in meal.items]

    for item in req.new_items:
        if item.weight and item.weight > 0:
            nut = estimate_nutrition(item.name, float(item.weight))
            items.append(
                ItemOut(
                    name=item.name,
                    weight=float(item.weight),
                    calories=coerce_number(nut.get("calories", 0)),
                    protein=coerce_number(nut.get("protein", 0)),
                    carbs=coerce_number(nut.get("carbs", 0)),
                    fats=coerce_number(nut.get("fats", 0)),
                    ingredients=[],
                )
            )

    updated = AnalysisResponse(
        id=meal.id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        image_meta=meal.image_meta,
        items=items,
        total=sum_totals(items),
        confidence_score=meal.confidence_score,
    )
    save_recent(updated)
    return updated


@app.get("/meals/recent")
async def get_recent_meals(limit: int = Query(10, ge=1, le=50)):
    return {"meals": get_recent(limit)}


# Optional: Health check
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test:app", host="0.0.0.0", port=8000)
