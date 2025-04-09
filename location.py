import requests
from config import settings
from db import SessionLocal
from sqlalchemy import select
from models import Memorial

API_KEY = settings.GOOGLE_MAPS_API_KEY

def get_formatted_location(latitude, longitude):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={API_KEY}&language=ja"
    response = requests.get(url)
    data = response.json()

    if "results" in data and len(data["results"]) > 0:
        address_components = data["results"][0]["address_components"]
        country = prefecture = city = ""
        for component in address_components:
            if "country" in component["types"]:
                country = component["long_name"]
            elif "administrative_area_level_1" in component["types"]:
                prefecture = component["long_name"]
            elif "locality" in component["types"] or "administrative_area_level_2" in component["types"]:
                city = component["long_name"]
        if country == "日本":
            return f"{prefecture} {city}".replace("都", "").replace("府", "").replace("県", "")
        else:
            return f"{country} {city}"
    return "住所情報が取得できませんでした"

def update_location_data():
    db = SessionLocal()
    updated = 0

    try:
        stmt = select(Memorial).where(
            (Memorial.latitude.isnot(None)) &
            (Memorial.longitude.isnot(None)) &
            ((Memorial.location.is_(None)) | (Memorial.location == ""))
        )
        memorials = db.execute(stmt).scalars().all()

        for m in memorials:
            location = get_formatted_location(m.latitude, m.longitude)
            m.location = location
            updated += 1
            print(f"✅ ID {m.memorials_id} → {location}")

        db.commit()
        print(f"🎉 更新完了（{updated}件）")
        return f"更新完了 ✅（{updated} 件）"

    except Exception as e:
        db.rollback()
        print(f"❌ エラー: {e}")
        return f"エラー: {e}"

    finally:
        db.close()
