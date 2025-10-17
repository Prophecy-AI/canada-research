#!/usr/bin/env python3
"""
Script to help accept Kaggle competition rules for all competitions in a set.

This script will:
1. Check which competitions you've already accepted
2. Open unaccepted competitions' rules pages in your browser
3. Verify acceptance after you click "I Understand and Accept"

Usage:
    python scripts/accept_kaggle_rules.py [competition-set-file]

Example:
    python scripts/accept_kaggle_rules.py experiments/splits/custom-set.txt
"""

import argparse
import sys
import time
import webbrowser
from pathlib import Path

# Add mlebench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlebench.utils import authenticate_kaggle_api, get_logger

logger = get_logger(__name__)


def read_competition_set(filepath: Path) -> list[str]:
    """Read competition IDs from a file, skipping comments and empty lines."""
    competitions = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                competitions.append(line)
    return competitions


def check_competition_accepted(api, competition_id: str) -> bool:
    """
    Check if competition rules have been accepted.

    Returns True if accepted, False if not accepted, None if can't determine.
    """
    from kaggle.rest import ApiException

    try:
        # Try to download competition files (will fail if rules not accepted)
        # Using a dry-run approach: check competition details first
        competition = api.competition_view(competition_id)

        # If we can view the competition, check if we can list files
        try:
            files = api.competition_list_files(competition_id)
            # If we can list files, rules are likely accepted
            return True
        except ApiException as e:
            if "accept" in str(e).lower() and "rules" in str(e).lower():
                return False
            # Other error, can't determine
            logger.warning(f"Could not determine acceptance status for {competition_id}: {e}")
            return None
    except ApiException as e:
        # Competition doesn't exist or other error
        logger.warning(f"Error checking {competition_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Accept Kaggle competition rules for multiple competitions"
    )
    parser.add_argument(
        "competition_set",
        nargs="?",
        default="experiments/splits/custom-set.txt",
        help="Path to competition set file (default: experiments/splits/custom-set.txt)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check acceptance status, don't open browser",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Open all competitions, even if already accepted",
    )
    args = parser.parse_args()

    competition_set_path = Path(args.competition_set)

    if not competition_set_path.exists():
        print(f"❌ ERROR: Competition set file not found: {competition_set_path}")
        print()
        print("Usage:")
        print("  python scripts/accept_kaggle_rules.py [competition-set-file]")
        print()
        print("Example:")
        print("  python scripts/accept_kaggle_rules.py experiments/splits/custom-set.txt")
        sys.exit(1)

    # Read competitions
    competitions = read_competition_set(competition_set_path)

    if not competitions:
        print(f"❌ ERROR: No competitions found in {competition_set_path}")
        sys.exit(1)

    print("=" * 50)
    print("Kaggle Competition Rules Accepter")
    print("=" * 50)
    print(f"Competition set: {competition_set_path}")
    print(f"Total competitions: {len(competitions)}")
    print()

    # Authenticate with Kaggle
    print("Authenticating with Kaggle API...")
    try:
        api = authenticate_kaggle_api()
        print("✅ Authenticated successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to authenticate with Kaggle API: {e}")
        print()
        print("Make sure you have ~/.kaggle/kaggle.json configured.")
        print("See: https://github.com/Kaggle/kaggle-api#api-credentials")
        sys.exit(1)

    print()

    # Check acceptance status
    print("Checking competition acceptance status...")
    print()

    accepted = []
    not_accepted = []
    unknown = []

    for i, comp_id in enumerate(competitions, 1):
        status_str = f"[{i}/{len(competitions)}] {comp_id}"
        print(f"{status_str}...", end="", flush=True)

        status = check_competition_accepted(api, comp_id)

        if status is True:
            print(" ✅ Accepted")
            accepted.append(comp_id)
        elif status is False:
            print(" ❌ Not accepted")
            not_accepted.append(comp_id)
        else:
            print(" ⚠️  Unknown")
            unknown.append(comp_id)

        # Rate limit: don't hammer the API
        time.sleep(0.5)

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"✅ Already accepted: {len(accepted)}")
    print(f"❌ Not accepted: {len(not_accepted)}")
    print(f"⚠️  Unknown status: {len(unknown)}")
    print()

    if args.check_only:
        if not_accepted:
            print("Competitions that need acceptance:")
            for comp_id in not_accepted:
                print(f"  - {comp_id}")
                print(f"    https://www.kaggle.com/c/{comp_id}/rules")
        print()
        print("Run without --check-only to open rules pages in browser.")
        sys.exit(0)

    # Determine which competitions to process
    to_process = []
    if args.force:
        to_process = competitions
        print("--force mode: will open all competitions")
    else:
        to_process = not_accepted + unknown
        if not to_process:
            print("✅ All competitions already accepted!")
            print()
            print("You can now run:")
            print("  cd mle-bench")
            print("  ./RUN_AGENT_V5_KAGGLE.sh")
            sys.exit(0)

    print(f"Will process {len(to_process)} competitions")
    print()
    print("For each competition:")
    print("  1. Rules page will open in your browser")
    print("  2. Scroll to bottom and click 'I Understand and Accept'")
    print("  3. Return to terminal and press Enter")
    print()

    response = input("Press Enter to start (or Ctrl+C to cancel)...")
    print()

    # Process each competition
    for i, comp_id in enumerate(to_process, 1):
        print("=" * 50)
        print(f"[{i}/{len(to_process)}] {comp_id}")
        print("=" * 50)

        rules_url = f"https://www.kaggle.com/c/{comp_id}/rules"

        print(f"Opening: {rules_url}")
        print()

        # Open in browser
        try:
            webbrowser.open(rules_url)
            print("✅ Opened in browser")
        except Exception as e:
            print(f"⚠️  Could not auto-open browser: {e}")
            print(f"   Please manually visit: {rules_url}")

        print()
        print("Instructions:")
        print("  1. Scroll to bottom of the page")
        print("  2. Click 'I Understand and Accept'")
        print("  3. Return here and press Enter to continue")
        print()
        print("(If already accepted, just press Enter to skip)")

        input("Press Enter after accepting rules...")

        # Verify acceptance
        print("Verifying acceptance...", end="", flush=True)
        time.sleep(1)  # Give Kaggle API time to update
        status = check_competition_accepted(api, comp_id)

        if status is True:
            print(" ✅ Verified!")
        elif status is False:
            print(" ⚠️  Still not accepted (may take a moment to sync)")
        else:
            print(" ⚠️  Could not verify")

        print()

        # Small delay to avoid overwhelming browser
        time.sleep(1)

    print("=" * 50)
    print("✅ COMPLETE!")
    print("=" * 50)
    print(f"Processed {len(to_process)} competitions")
    print()
    print("You can now run:")
    print("  cd mle-bench")
    print("  ./RUN_AGENT_V5_KAGGLE.sh")
    print()
    print("Or prepare datasets manually:")
    print(f"  mlebench prepare --competition-set {competition_set_path}")
    print()


if __name__ == "__main__":
    main()
