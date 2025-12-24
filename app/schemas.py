from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field

class WinkerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: Optional[str] = None
    email: Optional[str] = None
    photoProfil: Optional[str] = None
    sexe: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    subregion: Optional[str] = None
    pays: Optional[str] = None
    codePostal: Optional[str] = None
    lon: Optional[float] = None
    lat: Optional[float] = None
    currentLangue: Optional[str] = None

class WinkerIn(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: Optional[str] = None
    email: Optional[str] = None
    photoProfil: Optional[str] = None
    sexe: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    subregion: Optional[str] = None
    pays: Optional[str] = None
    codePostal: Optional[str] = None
    lon: Optional[float] = None
    lat: Optional[float] = None
    currentLangue: Optional[str] = None

class ParticipeWinkerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    rang: Optional[int] = None
    nbUnseen: Optional[int] = None
    dateOrder: Optional[str] = None
    participeWinker: Optional[WinkerOut] = None


class FilesEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    image: Optional[str] = None
    video: Optional[str] = None
    event_id: Optional[int] = None


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    creatorWinker: Optional[WinkerOut] = None
    participants: List[ParticipeWinkerOut] = []

    filesEvent: List[FilesEventOut] = Field(default_factory=list)  # ✅ AJOUTE ÇA


    # champs Event (reprend la liste de ton serializer, tu peux compléter au fur et à mesure)
    titre: Optional[str] = None
    datePublication: Optional[str] = None
    adresse: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    subregion: Optional[str] = None
    pays: Optional[str] = None
    codePostal: Optional[str] = None

    ageMinimum: Optional[int] = None
    ageMaximum: Optional[int] = None

    accessOuvert: Optional[bool] = None
    accessTous: Optional[bool] = None
    accessFille: Optional[bool] = None
    accessGarcon: Optional[bool] = None
    accessFollow: Optional[bool] = None
    accessFollower: Optional[bool] = None

    bioEvent: Optional[str] = None
    bioEvent_fr: Optional[str] = None
    titre_fr: Optional[str] = None

    moyenneAge: Optional[int] = None
    nbFille: Optional[int] = None
    nbGarcon: Optional[int] = None
    nbComment: Optional[int] = None
    numberView: Optional[int] = None

    lon: Optional[float] = None
    lat: Optional[float] = None

    detailsAddress: Optional[str] = None
    accessComment: Optional[bool] = None

    containReduction: Optional[bool] = None
    prixInitial: Optional[float] = None
    prixReduction: Optional[float] = None
    textReduction: Optional[str] = None

    needReservation: Optional[str] = None
    linkReservation: Optional[str] = None

    currentNbParticipants: Optional[int] = None
    maxNumberParticipant: Optional[int] = None
    isFull: Optional[bool] = None

    website: Optional[str] = None
    urlGoogleMapsAvis: Optional[str] = None
    urlAjoutGoogleMapsAvis: Optional[str] = None
    nb_conversations: Optional[int] = None
    nbStories: Optional[int] = None


class EventIn(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    creatorWinker: Optional[WinkerOut] = None
    participants: List[ParticipeWinkerOut] = []

    # champs Event (reprend la liste de ton serializer, tu peux compléter au fur et à mesure)
    titre: Optional[str] = None
    datePublication: Optional[str] = None
    adresse: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    subregion: Optional[str] = None
    pays: Optional[str] = None
    codePostal: Optional[str] = None

    ageMinimum: Optional[int] = None
    ageMaximum: Optional[int] = None

    accessOuvert: Optional[bool] = None
    accessTous: Optional[bool] = None
    accessFille: Optional[bool] = None
    accessGarcon: Optional[bool] = None
    accessFollow: Optional[bool] = None
    accessFollower: Optional[bool] = None

    bioEvent: Optional[str] = None
    bioEvent_fr: Optional[str] = None
    titre_fr: Optional[str] = None

    moyenneAge: Optional[int] = None
    nbFille: Optional[int] = None
    nbGarcon: Optional[int] = None
    nbComment: Optional[int] = None
    numberView: Optional[int] = None

    lon: Optional[float] = None
    lat: Optional[float] = None

    detailsAddress: Optional[str] = None
    accessComment: Optional[bool] = None

    containReduction: Optional[bool] = None
    prixInitial: Optional[float] = None
    prixReduction: Optional[float] = None
    textReduction: Optional[str] = None

    needReservation: Optional[str] = None
    linkReservation: Optional[str] = None

    currentNbParticipants: Optional[int] = None
    maxNumberParticipant: Optional[int] = None
    isFull: Optional[bool] = None

    website: Optional[str] = None
    urlGoogleMapsAvis: Optional[str] = None
    urlAjoutGoogleMapsAvis: Optional[str] = None
    nb_conversations: Optional[int] = None
    nbStories: Optional[int] = None


