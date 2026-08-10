"""Generate a reproducible bilingual demonstration dataset.

This data makes the application runnable, but it is not a substitute for a
company's labelled complaints. Metrics produced from it are explicitly marked
as demonstration metrics in the interface.
"""

from __future__ import annotations

from pathlib import Path
import random

import pandas as pd


TEMPLATES = {
    "Delivery": {
        "fr": [
            "Ma commande n'est toujours pas arrivée",
            "Le colis est arrivé avec beaucoup de retard",
            "Le livreur n'a pas trouvé mon adresse",
            "La date de livraison promise est dépassée",
            "Le suivi indique livré mais je n'ai rien reçu",
            "La livraison a été envoyée à une mauvaise adresse",
        ],
        "ar": [
            "الطلبية متاعي مازالت ما وصلتش",
            "الطرد وصل متأخر برشة",
            "الموزع ما لقاش العنوان متاعي",
            "فات موعد التوصيل المتفق عليه",
            "التتبع يقول تم التسليم اما ما استلمت شي",
            "بعثوا الطلبية لعنوان غالط",
        ],
    },
    "Damaged product": {
        "fr": [
            "Le produit est arrivé cassé",
            "L'emballage était ouvert et l'article abîmé",
            "Il manque une pièce dans le carton",
            "Le produit présente des rayures importantes",
            "L'article reçu est inutilisable à cause des dégâts",
            "Le colis était écrasé pendant le transport",
        ],
        "ar": [
            "المنتوج وصل مكسّر",
            "الكرتونة محلولة والسلعة مضروبة",
            "فما قطعة ناقصة في العلبة",
            "المنتوج فيه خدوش كبار",
            "السلعة ما تتستعملش من كثرة الضرر",
            "الطرد تفسخ في النقل",
        ],
    },
    "Payment": {
        "fr": [
            "Ma carte a été débitée deux fois",
            "Le paiement est refusé sans explication",
            "Le montant facturé est incorrect",
            "Je ne peux pas terminer le paiement en ligne",
            "Des frais inconnus apparaissent sur ma facture",
            "Le reçu de paiement n'est jamais arrivé",
        ],
        "ar": [
            "خصموا من البطاقة مرتين",
            "الدفع مرفوض من غير تفسير",
            "المبلغ في الفاتورة غالط",
            "ما نجمتش نكمل الخلاص على الموقع",
            "فما مصاريف ما نعرفهاش في الفاتورة",
            "وصل الخلاص ما وصلنيش",
        ],
    },
    "Customer service": {
        "fr": [
            "Le service client ne répond jamais",
            "Le conseiller était impoli et inutile",
            "J'ai été transféré plusieurs fois sans solution",
            "Personne ne rappelle malgré mes demandes",
            "L'agent a fermé mon dossier sans explication",
            "L'attente au téléphone est beaucoup trop longue",
        ],
        "ar": [
            "خدمة الحرفاء ما يجاوبوش",
            "المستشار كلامه خايب وما عاوننيش",
            "عداوني بين برشة أعوان من غير حل",
            "حتى حد ما رجع كلمني",
            "العميل سكر الملف من غير تفسير",
            "الانتظار في التلفون طويل برشة",
        ],
    },
    "Technical problem": {
        "fr": [
            "L'application se ferme dès son ouverture",
            "Je ne peux plus me connecter à mon compte",
            "Le site affiche une erreur à chaque tentative",
            "La connexion internet coupe sans arrêt",
            "Le code de vérification ne fonctionne pas",
            "La dernière mise à jour a bloqué le service",
        ],
        "ar": [
            "التطبيق يتسكر كي نحلّه",
            "ما عادش نجم ندخل للحساب",
            "الموقع يعطيني erreur كل مرة",
            "الانترنت تقص كل شوية",
            "كود التفعيل ما يخدمش",
            "آخر تحديث وقف الخدمة",
        ],
    },
    "Refund request": {
        "fr": [
            "Je demande le remboursement de ma commande",
            "Mon remboursement n'est toujours pas arrivé",
            "Je veux annuler et récupérer mon argent",
            "Le délai de remboursement est dépassé",
            "Remboursez le montant prélevé par erreur",
            "Ma demande de remboursement a été ignorée",
        ],
        "ar": [
            "نطلب ترجيع فلوس الطلبية",
            "فلوس الاسترجاع مازالت ما وصلتش",
            "نحب نلغي ونسترجع فلوسي",
            "فات أجل استرجاع المبلغ",
            "رجعولي المبلغ اللي خصمتوه بالغلط",
            "طلب استرجاع الفلوس تجاهلوه",
        ],
    },
}

TONES = {
    "fr": {
        "negative": ["C'est inadmissible.", "Je suis très déçu.", "Cette situation me met en colère."],
        "neutral": ["Merci de vérifier le dossier.", "Je souhaite connaître la situation.", "Pouvez-vous examiner ce cas ?"],
        "positive": ["Merci pour votre aide habituelle.", "Je reste satisfait mais merci de corriger cela.", "Votre équipe aide bien, j'attends une solution."],
    },
    "ar": {
        "negative": ["هذا غير مقبول.", "انا مستاء برشة.", "الوضعية هاذي قلقتني."],
        "neutral": ["يرجى التثبت من الملف.", "نحب نعرف شنو صار.", "تنجموا تشوفوا الحالة؟"],
        "positive": ["شكرا على تعاونكم المعتاد.", "راضي على خدمتكم اما اصلحوا المشكلة.", "فريقكم يعاون ونستنى في الحل."],
    },
}

URGENCY = {
    "fr": {
        "low": ["Ce n'est pas urgent.", "Vous pouvez répondre cette semaine."],
        "medium": ["J'attends une réponse rapidement.", "Merci de traiter cela bientôt."],
        "high": ["Urgent, j'en ai besoin aujourd'hui !", "Résolvez cela immédiatement, activité bloquée !"],
    },
    "ar": {
        "low": ["الموضوع موش مستعجل.", "تنجموا تجاوبوني الأسبوع هذا."],
        "medium": ["نستنى في جواب قريب.", "يرجى المعالجة في أقرب وقت."],
        "high": ["استعجالي، يلزمني حل اليوم!", "حلّوها توّا، الخدمة واقفة!"],
    },
}


def generate_sample_data(path: str | Path | None = None, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    products = ["Mobile App", "Internet Box", "Marketplace", "Payment Card"]
    locations = ["Tunis", "Sfax", "Sousse", "Bizerte", "Gabès"]
    start = pd.Timestamp("2026-01-01")
    counter = 1
    for topic, languages in TEMPLATES.items():
        for language, phrases in languages.items():
            for repetition in range(3):
                for index, phrase in enumerate(phrases):
                    sentiment = ["negative", "neutral", "positive"][(index + repetition) % 3]
                    urgency = ["low", "medium", "high"][(index + 2 * repetition) % 3]
                    text = f"{phrase}. {rng.choice(TONES[language][sentiment])} {rng.choice(URGENCY[language][urgency])}"
                    rows.append(
                        {
                            "complaint_id": f"CMP-{counter:04d}",
                            "date": start + pd.Timedelta(days=rng.randrange(180)),
                            "text": text,
                            "language": language,
                            "topic": topic,
                            "sentiment": sentiment,
                            "urgency": urgency,
                            "product": rng.choice(products),
                            "location": rng.choice(locations),
                        }
                    )
                    counter += 1
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df

