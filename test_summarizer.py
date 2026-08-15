"""
Quick smoke test for hf_summarizer.py
Run: python test_summarizer.py
"""

from app.summarizer.hf_summarizer import summarize

sample = (
    "Patient is a 58-year-old male presenting with a 3-week history of "
    "progressively worsening chest pain radiating to the left arm, accompanied "
    "by shortness of breath and diaphoresis. ECG findings show ST-segment "
    "elevation in leads II, III, and aVF consistent with inferior STEMI. "
    "Troponin I levels are markedly elevated at 12.4 ng/mL. Patient has a history "
    "of type 2 diabetes mellitus, hypertension, and hyperlipidemia. Current "
    "medications include metformin 1000mg BD, amlodipine 5mg OD, and atorvastatin "
    "40mg OD. Patient was taken for emergency PCI; drug-eluting stent placed in "
    "right coronary artery with TIMI 3 flow restored. Post-procedure vitals are "
    "stable. Echocardiogram shows EF of 45% with inferior wall hypokinesis. Plan "
    "includes dual antiplatelet therapy, beta-blocker, ACE inhibitor, and cardiac "
    "rehabilitation referral."
)

print("Running HuggingFace summarizer...")
print("(First run will download the model -- this may take a minute)\n")

result = summarize(sample)

print("=" * 60)
print("SUMMARY:")
print("=" * 60)
print(result)
