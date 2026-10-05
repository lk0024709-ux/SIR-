"""Build SIR's self-authored pipeline-validation fixture corpus.

This is the ONLY corpus the repository can legally ship and use offline, so it is written by
hand (CC0) rather than scraped. Read the honesty constraints before using any number derived
from it:

  * Documents are COMPOSED from a seed sentence set with a seeded RNG. They have realistic
    structure (topics, paragraphs, document boundaries, deliberate near-duplicates and dirty
    markup so the cleaning/dedup stages are actually exercised) but they are NOT natural text.
  * Therefore every metric computed only on this fixture is labelled `provisional` by the
    tokenizer/LM scripts and must never be published as a real-world Hindi/Hinglish capability.
  * Output is committed as ``docs.json`` (not ``.jsonl``) because ``data/**/*.jsonl`` is
    git-ignored on purpose: real corpora must never land in git.

Deterministic by construction: same ``--seed`` and same script version => byte-identical
output, which tests rely on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "fixtures" / "sir_fixture_v0"
FIXTURE_ID = "sir_fixture_v0"
GENERATOR = "scripts/build_fixture_corpus.py"

# ======================================================================================
# Seed sentences (authored for SIR, CC0). Grouped by topic so composed documents have
# topical coherence, which matters because dedup/near-dup and per-topic probes both care.
# ======================================================================================
HI: dict[str, list[str]] = {
    "vigyan": [
        "विक्रम साराभाई ने अंतरिक्ष कार्यक्रम को विकास से जोड़कर देखा था।",
        "भारत का मंगल अभियान प्रथम प्रयास में ही मंगल कक्षा तक पहुँच गया था।",
        "रुचिका खन्ना अपनी प्रयोगशाला में सिलिकॉन सेंसर पर काम करती हैं।",
        "मौसम विभाग ने अगले चार दिनों तक भारी बारिश की चेतावनी जारी की है।",
        "राजस्थान में सोलर प्लांट की लागत पिछले दशक में काफी घटी है।",
        "समुद्र तट पर स्थापित यंत्र लहरों की ऊँचाई हर दस मिनट में दर्ज करते हैं।",
        "शोध पत्र में प्रयोग के आँकड़े और डेटा सेट दोनों सार्वजनिक करने चाहिए।",
        "नील ने परीक्षा से पहले अपना पूना नोट्स दोबारा पढ़े।",
        "इसरो के अनुसार मौसम के उपग्रह हर छह घंटे में छवि अपडेट करते हैं।",
        "प्रयोगशाला में तापमान नियंत्रण बिगड़ने से पूरा नमूना नष्ट हो गया।",
        "विज्ञान संपादक कहते हैं कि नकारात्मक परिणाम भी प्रकाशित होने चाहिए।",
        "जैव विविधता परीक्षण के लिए जिले में चार नए स्टेशन खोले गए हैं।",
    ],
    "shiksha": [
        "जिले के पंचायत स्कूलों में गणित की कक्षाएँ सुबह सात बजे से शुरू होती हैं।",
        "शिक्षिका ने बच्चों को स्थानीय भाषा में समझाया तो अवधारणा जल्दी पकड़ में आई।",
        "परीक्षा परिणाम के बाद स्कूल प्रबंधन समिति ने बैठक बुलाई।",
        "छात्रवृत्ति के लिए ऑनलाइन फॉर्म भरने की अंतिम तिथि बढ़ा दी गई है।",
        "पुस्तकालय में हिंदी, अंग्रेज़ी और संस्कृत की पांडुलिपियाँ रखी हैं।",
        "संस्कृत व्याकरण में संधि और समास दो अलग प्रक्रियाएँ हैं।",
        "विश्वविद्यालय ने शोध छात्रों के लिए प्रयोगशाला समय बढ़ाया है।",
        "अध्यापक प्रशिक्षण संस्थान में मूल्यांकन के नए मानक अपनाए गए हैं।",
        "माता-पिता ने कहा कि घर में पढ़ाई का समय तय होना चाहिए।",
        "बोर्ड ने पाठ्यक्रम से दो अध्याय हटाकर व्यावहारिक कार्य जोड़ा है।",
    ],
    "naagarik": [
        "नगर निगम ने जल भराव वाली सड़कों पर पंप लगाने का आदेश दिया है।",
        "आरटीआई के तहत माँगी गई जानकारी तीस दिन में देना अनिवार्य है।",
        "ग्राम पंचायत में सड़क मरम्मत के लिए अनुमोदित धनराशि घोषित की गई।",
        "अनुसूचित क्षेत्रों में शिक्षा अधिकारी की नियुक्ति राज्य सूची से होती है।",
        "उपभोक्ता मंच ने टोल फीस बिना रसीद वसूलने पर जुर्माना लगाया।",
        "शहर में बस रूट बदलने से यात्रियों को परेशानी हुई, इसलिए समीक्षा हो रही है।",
        "अस्पताल में शिकायत निवारण केंद्र का बोर्ड प्रवेश द्वार पर लगा है।",
        "मतदाता सूची में नाम जोड़ने के लिए प्रमाण पत्र आवश्यक है।",
        "नगर निगम आयुक्त ने कचरा पृथक्करण के नियम फिर से समझाए।",
        "शाखा पुस्तकालय शाम छह बजे बंद होता है, लेकिन परीक्षा अवधि में खुला रहता है।",
    ],
    "kisan": [
        "किसान ने कहा कि सिंचाई की नई नहर से फसल का समय बदल गया है।",
        "मंडी में गेहूँ का भाव पिछले सप्ताह की तुलना में स्थिर रहा।",
        "मौसम के अनुसार बुवाई करने से खर्च घटता है, विभाग कहता है।",
        "बाग़ान ने इस साल आम की पैदावार कम होने का अनुमान जताया है।",
        "सहकारी समिति ने बीज और खाद का उधार तीन किस्तों में लौटाने की छूट दी।",
        "पशुपालन इकाई में टीकाकरण का शिविर लगाया गया।",
        "राज्य में ड्रिप सिंचाई लगाने पर सब्सिडी का दावा ऑनलाइन होता है।",
        "ठेकेदार ने मज़दूरी का बकाया अदा नहीं किया, इसलिए शिकायत दर्ज हुई।",
    ],
    "khel": [
        "स्थानीय स्टेडियम में अंडर-19 टूर्नामेंट का फाइनल शनिवार को खेला जाएगा।",
        "युवा तेज़ गेंदबाज़ ने पहली ही गेंद पर विकेट लिया।",
        "मैराथन की तैयारी में कोच ने हर हफ़्ते की दूरी धीरे-धीरे बढ़ाई।",
        "लड़कियों के हॉकी दल ने तीन गोल करके सेमीफाइनल में जगह बनाई।",
        "खिलाड़ी की चोट की रिपोर्ट मेडिकल बोर्ड को भेज दी गई है।",
        "जिले के खेल अधिकारी ने कोर्ट की मरम्मत के लिए फंड माँगा।",
        "यूनिफॉर्म वितरण समारोह में स्कूल के प्रधानाचार्य मुख्य अतिथि थे।",
        "बारिश के कारण मैच दो घंटे देर से शुरू हुआ।",
    ],
    "tech": [
        "सॉफ़्टवेयर टीम ने बग ठीक करने के बाद परीक्षण रिपोर्ट फिर से चलाई।",
        "सर्वर पर लॉग बढ़ने से डिस्क भर गई, इसलिए रोटेशन नीति बदली गई।",
        "उपयोगकर्ता ने ऐप में लॉगिन करते समय ओटीपी की अवधि समाप्त होने की शिकायत की।",
        "नए फ़ीचर को रोलआउट करने से पहले दोपहरी की जगह रात में परीक्षण किया जाता है।",
        "डेटाबेस में अनुक्रमणिका जोड़ने से क्वेरी का समय चौथाई रह गया।",
        "परिक्षक ने रिपोर्ट किया कि अनधिकृत स्क्रिप्ट फ़ाइल अपलोड की जा सकती है।",
        "दस्तावेज़ में लिखा है कि एपीआई कुंजी क्लाइंट कोड में कभी नहीं रखनी चाहिए।",
        "मोबाइल संस्करण में ऑफ़लाइन मोड के लिए संग्रहण नीति तय करनी होगी।",
    ],
}

EN: dict[str, list[str]] = {
    "science": [
        "The laboratory recorded sensor drift over six weeks before it reported the result.",
        "A negative finding published honestly is worth more than a positive one that cannot be reproduced.",
        "The team measured latency on a laptop with the cooling fan disabled to expose thermal limits.",
        "Peer reviewers asked for the exact random seed and the full configuration file.",
        "The instrument was calibrated against a reference standard kept in a temperature-controlled room.",
        "Their report states the uncertainty range rather than a single confident number.",
        "The dataset was split before any filtering, so that cleaning could not leak labels.",
        "Field notes from the first expedition were digitised forty years later.",
    ],
    "civic": [
        "The municipal corporation published the tender documents in an open, machine-readable format.",
        "Residents filed a right-to-information request for the road repair expenditure statement.",
        "The ward officer said drainage work would finish before the monsoon, and the schedule was attached.",
        "A public hearing on the new bus route drew ninety attendees, most of them commuters.",
        "The court ordered the department to reply to the consumer complaint within four weeks.",
        "Voter list corrections were accepted online until the end of the month.",
        "The hospital's grievance desk logged two hundred calls in a week, mostly about appointments.",
    ],
    "education": [
        "Teachers reported that explaining a concept in the students' home language improved retention.",
        "The library's manuscript section is catalogued, but the scan quality is uneven.",
        "Examiners noted that a syllabus change left two cohorts without past papers.",
        "The scholarship portal crashed on the deadline, and the authority extended it by a week.",
        "Tutoring hours were reallocated from rote revision to practical problem solving.",
        "The university added an elective on statistics because social-science students needed it.",
    ],
    "technology": [
        "The service returned a partial result instead of failing, and the bug went unnoticed for days.",
        "Access logs showed a shared administrator password, so the team moved to individual keys.",
        "Disk usage climbed every night because the log rotation policy had never been enabled.",
        "The reviewer asked whether the model was evaluated on held-out data or on its own training text.",
        "A quantised build ran twice as fast but produced different numbers on the boundary cases.",
        "The migration script was idempotent, which is why re-running it caused no harm.",
        "Documentation listed the unsupported device class rather than pretending it worked.",
        "They committed the failing test first, then the fix, so the regression could not return silently.",
    ],
    "general": [
        "Rain arrived early and the market stalls were packed by eight in the morning.",
        "The marathon route was shortened after the heat advisory, and runners were notified by message.",
        "Her notebook had one rule: never write a number you cannot point a source at.",
        "The shopkeeper kept the receipts in a box, which made the refund straightforward.",
        "He read the contract twice before noticing the clause about automatic renewal.",
    ],
}

# Hinglish = genuine code-mixed Hindi-English as it is typed, not a transliteration of Hindi.
HINGLISH: dict[str, list[str]] = {
    "daily": [
        "aaj office nahi jaunga, meeting cancel ho gayi thi subah hi.",
        "boss ne bola hai ki report kal tak submit kar do, warna Friday review me problem hogi.",
        "garmi itni hai ki AC bhi cool nahi kar raha, technician ko bula lena chahiye.",
        "mummy boli shaar ko ghar aana, rishtedaar aa rahe hain.",
        "yaar recharge karna bhool gaya, isliye call cut ho gayi beech me.",
        "main soch raha hoon ki weekend pe Goa plan karein, but budget thoda tight hai.",
        "khana kha ke baad hi gym jana hai, warna pet me dard hoga.",
    ],
    "tech": [
        "model train karte waqt loss NaN aa gaya, shayad learning rate zyada hai.",
        "git push karne se pehle tests chalana chahiye tha, ab CI fail ho raha hai.",
        "server pe OOM kill aa raha hai, memory leak check karo bhai.",
        "dataset ki license verify nahi ki thi, isliye weights release nahi kar pa rahe.",
        "tokenizer Hindi me double tokens bana raha hai, vocab size badhana padega.",
        "Colab ka session disconnect ho gaya, checkpoint save karna zaroori tha.",
        "GPU nahi hai to CPU pe chhota model try karo, 25M params se upar mat jao.",
        "log me error nahi aa raha, par response slow hai - profiling karni padegi.",
    ],
    "study": [
        "physics ka numerical samajh nahi aaya, solution dekh ke copy kar liya.",
        "coach ne bola hai ki previous year papers pehle solve karo, mock baad me.",
        "history me dates confuse ho jaati hain, timeline bana ke yaad karna chahiye.",
        "english essay me grammar check karwa lena, marks wahi kat-te hain.",
        "assignment deadline Tuesday hai, Sunday ko start kiya to tension hogi.",
        "group study me sab phone chala rahe the, individually padhna better tha.",
    ],
    "money": [
        "EMI ki date miss ho gayi, penalty lag gaya do sau rupaye.",
        "UPI payment fail hua lekin paisa kat gaya, bank me complaint dalni padegi.",
        "rent increase ho gaya hai, landlord naya agreement bhej raha hai.",
        "sale me laptop sasta mil raha tha par EMI option nahi tha.",
        "budget me bachat ka hisaab karte waqt pata chala ki last month extra gaya hai.",
    ],
    "travel": [
        "train me confirmation nahi mila, waitlist 47 tha, isliye bus se jaana padega.",
        "platform change ho gaya, announcement late hui thi station pe.",
        "flight delay hui to airline ne hotel voucher diya, claim karna easy tha.",
        "traffic ki wajah se 40 minute lage airport pahunchne me, next time early nikalunga.",
        "metro card me balance khatam ho gaya, gate pe line lag gayi.",
    ],
}

# Hinglish typed in Devanagari with Latin-script English words — how a large share of Indian
# technical chat is actually written, and the case an English-trained byte-level vocabulary handles
# worst (Devanagari costs 3 bytes/char while the English tokens are already merged).
HINGLISH_DEVA: dict[str, list[str]] = {
    "kaam": [
        "मैं अभी report finalize कर रहा हूँ, दस मिनट बाद भेजता हूँ।",
        "server restart करते ही training दोबारा शुरू हो गई, लेकिन checkpoint बचा नहीं था।",
        "यह bug सिर्फ़ Android 13 पर reproduce होता है, emulator पर नहीं।",
        "पहले dataset की license check करो, फिर weights release करना safe है।",
        "आज की meeting में GPU budget clear नहीं हुआ, इसलिए छोटे model पर जा रहे हैं।",
        "tokenizer की vocab size बढ़ाने से Hindi tokens 18% कम हो गए, यह number परीक्षण में आया।",
        "नया feature ship करने से पहले offline mode test करना मत भूलना।",
        "unit test पास हो गए, पर integration test में schema mismatch आ रहा है।",
        "क्वांटनाइज़ेशन के बाद perplexity 3% बढ़ी, trade-off स्वीकार्य है।",
        "नोट्स में लिखो कि कौन-सा seed इस्तेमाल हुआ, वरना result repeat नहीं होगा।",
        "प्रोडक्शन में जाने से पहले rollback plan तैयार रखना ज़रूरी है।",
        "आज का progress: pipeline चल गया, पर corpus अभी licensed नहीं है।",
    ],
    "padhai": [
        "असाइनमेंट की deadline Tuesday है, Sunday से शुरू किया तो टेंशन होगी।",
        "physics के numerical में unit conversion गलत हुआ, इसलिए उत्तर आधा आया।",
        "previous year papers solve करने के बाद mock test देना चाहिए।",
        "essay में grammar की गलतियाँ अंक घटाती हैं, proofread ज़रूरी है।",
        "group study में सब phone चला रहे थे, अकेले पढ़ना बेहतर रहा।",
        "syllabus के दो अध्याय हटने से past papers का उपयोग कम हो गया।",
        "library में मिली पुरानी पांडुलिपि का scan बहुत धुंधला निकला।",
        "lab report में uncertainty लिखना अनिवार्य है, केवल उत्तर नहीं।",
        "फ़ीस जमा करने का रसीद नंबर सुरक्षित रखो, बाद में काम आएगा।",
        "छात्रवृत्ति का फॉर्म भरने के लिए income certificate चाहिए।",
    ],
    "roz": [
        "गर्मियों में पानी की कमी हो जाती है, टंकी भरवाना ज़रूरी है।",
        "बाज़ार में आलू के दाम अचानक बढ़ गए, सब्ज़ीवाले ने बताया फ़सल खराब हुई।",
        "बस स्टैंड पर 20 मिनट खड़े रहने के बाद भी बस नहीं आई।",
        "रेलवे स्टेशन पर platform बदलने की सूचना देर से मिली।",
        "मोबाइल recharge में पैसा कट गया पर talktime नहीं मिला, शिकायत दर्ज करवाई।",
        "शहर में प्रदूषण बढ़ने से अस्पतालों में साँस की शिकायतें बढ़ी हैं।",
        "तापमान 44 डिग्री पहुँचा, मौसम विभाग ने लू की चेतावनी दी।",
        "गाँव में सौर पैनल लगने के बाद बिजली कटौती कम हो गई है।",
        "मंदिर के पास लगने वाले मेले में भीड़ बहुत ज़्यादा थी।",
        "रसोई में गैस खत्म हो गई, सिलिंडर बदलवाने में दो घंटे लगे।",
    ],
}

# Documents deliberately left "dirty" so the cleaning stage is not tested against clean-only text.
DIRTY_MARKUP = [
    "नगर निगम {{update}} ने जल भराव वाली सड़कों पर पंप लगाने का आदेश दिया <ref name=a1/>।",
    "The council approved the tender [citation needed] — see https://example.gov/notice for details.",
    "model train karte waqt loss NaN aa gaya\t\tdekh lo logs me",
    "à¤®à¥‡à¤‚ à¤—à¤°à¥€ à¤¬à¤¾à¤°à¤¿à¤¶ à¤¹à¥‹à¤—à¥€",  # mojibake: Devanagari mis-decoded as latin-1
    "युवा तेज़ गेंदबाज़​ने पहली ही गेंद पर विकेट लिया।",  # zero-width space inside
    "पुस्तकालय में हिंदी, अंग्रेज़ी और संस्कृत की पांडुलिपियाँ रखी हैं.\n\n\n\n   ",
    "<p>छात्रवृत्ति के लिए ऑनलाइन फॉर्म भरने की अंतिम तिथि बढ़ा दी गई है।</p>",
    "The lab recorded sensor drift. The lab recorded sensor drift. The lab recorded sensor drift.",
]

BUCKETS = {"hi": HI, "en": EN, "hinc-latn": HINGLISH, "hinc-deva": HINGLISH_DEVA}


def compose_corpus(seed: int, docs_per_bucket: int) -> tuple[list[dict], list[dict]]:
    """Return (documents, dirty_documents). Deterministic for a fixed seed."""
    rng = random.Random(seed)
    docs: list[dict] = []
    for lang, bucket in BUCKETS.items():
        for topic, sentences in bucket.items():
            for n in range(docs_per_bucket):
                # pick 3-7 sentences without replacement inside a doc (coherent, not random soup)
                k = min(len(sentences), rng.randint(3, 7))
                chosen = rng.sample(sentences, k)
                rng.shuffle(chosen)
                paras, cur = [], []
                for i, s in enumerate(chosen):
                    cur.append(s)
                    if len(cur) == rng.choice([2, 2, 3]) or i == len(chosen) - 1:
                        paras.append(" ".join(cur))
                        cur = []
                docs.append(
                    {
                        "id": f"fx-{lang}-{topic}-{n:04d}",
                        "source_id": FIXTURE_ID,
                        "language": lang,
                        "topic": topic,
                        "text": "\n\n".join(paras),
                    }
                )

    dirty = [
        {
            "id": f"fx-dirty-{i:03d}",
            "source_id": FIXTURE_ID,
            "language": ["hi", "en", "hinc-latn", "hinc-deva", "unknown"][i % 5],
            "topic": "dirty",
            "text": t,
            "note": "intentionally malformed: markup, mojibake, zero-width chars, repetition",
        }
        for i, t in enumerate(DIRTY_MARKUP)
    ]
    # inject exact + near duplicates so dedup stages have something to find
    dup_docs: list[dict] = []
    for i, d in enumerate(docs[: max(1, len(docs) // 10)]):
        if i % 3 == 0:  # exact duplicate under a new id
            dup_docs.append({**d, "id": d["id"] + "-exactdup"})
        elif i % 3 == 1:  # near duplicate: one char-level edit + reordered punctuation
            dup_docs.append({**d, "id": d["id"] + "-neardup", "text": d["text"].replace("।", ". ", 1)})
        else:  # duplicate with a sentence appended (should survive near-dup, caught by doc-level minhash)
            extra = "अतिरिक्त वाक्य जोड़ा गया है।" if d["language"] == "hi" else "One extra sentence was appended."
            dup_docs.append({**d, "id": d["id"] + "-extended", "text": d["text"] + " " + extra})
    return docs + dirty + dup_docs, dirty


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the self-authored SIR fixture corpus (CC0).")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--docs-per-bucket", type=int, default=26)
    ap.add_argument("--force", action="store_true", help="overwrite an existing fixture")
    args = ap.parse_args(argv)

    out = Path(args.out)
    target = out / "docs.json"
    if target.exists() and not args.force:
        print(f"fixture already exists: {target} (use --force to rebuild)")
    else:
        docs, dirty = compose_corpus(args.seed, args.docs_per_bucket)
        out.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(docs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

        sentences = sum(len(v) for b in BUCKETS.values() for v in b.values())
        chars = sum(len(d["text"]) for d in docs)
        payload = {
            "fixture_id": FIXTURE_ID,
            "license": "CC0-1.0",
            "authorship": "authored for the SIR repository; seed sentences are human-written, documents are composed",
            "generator": GENERATOR,
            "seed": args.seed,
            "docs_per_bucket": args.docs_per_bucket,
            "documents": len(docs),
            "dirty_documents": len(dirty),
            "seed_sentences": sentences,
            "chars_total": chars,
            "natural": False,
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "languages": {"hi": 0, "en": 0, "hinc-latn": 0, "hinc-deva": 0, "unknown": 0},
        }
        for d in docs:
            payload["languages"][d["language"]] = payload["languages"].get(d["language"], 0) + 1
        payload["sha256_docs_json"] = hashlib.sha256(target.read_bytes()).hexdigest()
        (out / "corpus_meta.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (out / "CARD.md").write_text(_card(payload), encoding="utf-8")

    meta = json.loads((out / "corpus_meta.json").read_text(encoding="utf-8"))
    print(f"wrote {target}")
    print(f"  documents={meta['documents']} chars={meta['chars_total']} seed={meta['seed']}")
    print(f"  sha256={meta['sha256_docs_json'][:16]}…  natural={meta['natural']}")
    print("  NOTE: composed fixture — metrics from it are pipeline-validation, not capability results.")
    return 0


def _card(m: dict) -> str:
    return f"""# Data card: `{m['fixture_id']}`

| Field | Value |
|-------|-------|
| License | {m['license']} |
| Authorship | {m['authorship']} |
| Natural corpus? | **No** — composed from {m['seed_sentences']} seed sentences by `{m['generator']}` (seed {m['seed']}) |
| Documents | {m['documents']} |
| Characters | {m['chars_total']} (measured) |
| Languages | {m['languages']} |
| Dirty/markup samples | {m['dirty_documents']} |
| sha256(docs.json) | `{m['sha256_docs_json']}` |
| Built | {m['built_at_utc']} |

## What it is for
Exercising every stage that a real corpus must pass — cleaning, dedup, near-dup, splitting,
leakage measurement, tokenizer bake-off plumbing, one-step and short training runs, and CPU
inference — **without network access and without any licensing risk**.

## What it is not
It is not a Hindi, English, or Hinglish corpus for training a model worth releasing. Documents
repeat a small sentence pool, so:

- token-efficiency numbers are biased toward whatever merges the tiny training slice supports;
- perplexity is not comparable to any external number;
- generation quality is meaningless as a capability statement.

Scripts therefore attach `provisional: true` to results computed on this fixture, and SIR's
README must never quote a number that came only from here as a capability claim.

## Deliberate flaws included (so the pipeline cannot pass by ignoring dirt)
markup/`{{{{update}}}}` templates, `<ref>`/`<p>` tags, bare URLs, mojibake (`à¤®à¥‡à¤‚ …`),
zero-width spaces inside words, CRLF/trailing whitespace, exact + near + extended duplicates,
and one repeated-sentence document.

## Rebuilding
```bash
python -m scripts.build_fixture_corpus --seed {m['seed']}   # byte-identical output
```
"""


if __name__ == "__main__":
    raise SystemExit(main())
