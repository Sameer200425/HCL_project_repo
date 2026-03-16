"""
CEDAR Signature Dataset Setup Instructions
==========================================
The CEDAR signature verification dataset requires manual download.

Dataset Information:
- Full name: CEDAR (Center of Excellence for Document Analysis and Recognition)
- Size: 2,640 signatures (55 signers × 24 genuine + 24 forged each)
- Use: Signature fraud detection, verification
- License: Research/academic use

DOWNLOAD STEPS:
===============
1. Visit: https://www.cedar.buffalo.edu/NIJ/data/
2. Fill out the data request form
3. Agree to the usage terms
4. Wait for approval (usually 1-3 business days)
5. Download the ZIP file when link is provided

After downloading, extract to:
    data/cedar_signatures/genuine/   -> original_*.png files
    data/cedar_signatures/forged/    -> forgeries_*.png files

ALTERNATIVE OPEN DATASETS:
==========================
If you need data immediately, consider these open-access alternatives:

1. GPDS Synthetic Signature Dataset
   - URL: https://www.gpds.ulpgc.es/
   - 4,000 signers, freely available for research

2. BHSig260 (Bengali/Hindi Signatures)
   - URL: https://github.com/signatured/BHSig260
   - 260 signers, open access

3. ICDAR 2011 SigComp
   - Competition dataset for signature verification
   - Available through ICDAR archives

CURRENT STATUS:
===============
Synthetic signatures have been generated for testing.
Replace with real CEDAR data for production use.
"""

from pathlib import Path

DATA_DIR = Path("data/cedar_signatures")

def check_cedar_status():
    """Check if CEDAR data has been downloaded."""
    genuine_dir = DATA_DIR / "genuine"
    forged_dir = DATA_DIR / "forged"
    
    genuine_count = len(list(genuine_dir.glob("*.png"))) if genuine_dir.exists() else 0
    forged_count = len(list(forged_dir.glob("*.png"))) if forged_dir.exists() else 0
    
    print("=" * 60)
    print("  CEDAR Signature Dataset Status")
    print("=" * 60)
    print(f"\nCurrent data in {DATA_DIR}:")
    print(f"  Genuine signatures: {genuine_count}")
    print(f"  Forged signatures:  {forged_count}")
    
    if genuine_count >= 1320 and forged_count >= 1320:
        print("\n✅ Full CEDAR dataset appears to be installed!")
        print("   Expected: 1,320 genuine + 1,320 forged = 2,640 total")
        return True
    elif genuine_count > 0 or forged_count > 0:
        print("\n⚠️  Partial data found (likely synthetic for testing)")
        print("   For production, download the full CEDAR dataset.")
    else:
        print("\n❌ No signature data found")
    
    print("\n" + "=" * 60)
    print("  DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print(__doc__)
    return False


if __name__ == "__main__":
    check_cedar_status()
