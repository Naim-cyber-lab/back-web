from app.schemas import *
from typing import Any, Dict
import json

def to_event_out(e: Dict[str, Any]) -> EventOut:
    creator = e.get("creatorWinker")
    files_raw = e.get("filesEvent") or []
    if isinstance(files_raw, str):  # au cas où psycopg renvoie du JSON en texte
        files_raw = json.loads(files_raw)

    files = [FilesEventOut.model_validate(x) for x in files_raw]
    return EventOut(
        id=e.get("id"),
        creatorWinker=WinkerOut.model_validate(creator) if creator else None,
        participants=[],  # tu as '[]'::jsonb côté SQL
        filesEvent=files,
        titre=e.get("titre"),
        datePublication=str(e.get("datePublication")) if e.get("datePublication") else None,
        adresse=e.get("adresse"),
        city=e.get("city"),
        region=e.get("region"),
        subregion=e.get("subregion"),
        pays=e.get("pays"),
        codePostal=e.get("codePostal"),

        bioEvent=e.get("bioEvent"),
        bioEvent_fr=e.get("bioEvent_fr"),
        titre_fr=e.get("titre_fr"),
        audio=e.get("audio"),

        lon=e.get("lon"),
        lat=e.get("lat"),

        accessComment=e.get("accessComment"),
        containReduction=e.get("containReduction"),
        prixInitial=e.get("prixInitial"),
        prixReduction=e.get("prixReduction"),

        firstPreference=e.get("firstPreference"),
        firstPhotoLocalisation=e.get("firstPhotoLocalisation"),
        remarque=e.get("remarque"),
        priceEvent=e.get("priceEvent"),
        horaireDebut=e.get("horaireDebut"),
        textForSearchingBar=e.get("textForSearchingBar"),
        vectorPreferenceEvent=e.get("vectorPreferenceEvent"),
        hastagEvents=e.get("hastagEvents"),
        textReduction=e.get("textReduction"),
        currentLangue=e.get("currentLangue"),
        importanteInformation=e.get("importanteInformation"),
        needReservation=e.get("needReservation"),
        linkReservation=e.get("linkReservation"),
        isFull=e.get("isFull"),
        detailsAddress=e.get("detailsAddress"),
        currentNbParticipants=e.get("currentNbParticipants"),
        maxNumberParticipant=e.get("maxNumberParticipant"),
        nbComment=e.get("nbComment"),
        website=e.get("website"),
        urlGoogleMapsAvis=e.get("urlGoogleMapsAvis"),
        urlAjoutGoogleMapsAvis=e.get("urlAjoutGoogleMapsAvis"),
        nb_conversations=e.get("nb_conversations"),
        nbStories=e.get("nbStories"),
    )


