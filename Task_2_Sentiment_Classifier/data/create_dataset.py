"""
Dataset Generator & Curation Script for Multi-Class Sentiment Analysis.

Generates a realistic, diverse, multi-domain dataset covering:
- Positive: Customer praise, stellar experiences, positive product reviews, satisfaction.
- Neutral: Objective facts, status updates, routine inquiries, confirmations, instructions.
- Negative: Service failures, technical bugs, frustration, complaints, cancellations.

Features natural language diversity, negation constructs, domain variations,
and edge cases to reflect realistic NLP benchmark conditions.
"""

import os
import random
import pandas as pd


def generate_curated_sentiment_dataset(output_path: str, samples_per_class: int = 400) -> pd.DataFrame:
    """
    Constructs a diverse, realistic 3-class sentiment dataset.

    Args:
        output_path: Target path to write CSV file.
        samples_per_class: Number of samples per sentiment class (default 400 => 1200 total).

    Returns:
        Pandas DataFrame containing 'text' and 'sentiment' columns.
    """
    random.seed(42)

    # Varied Positive phrases & clauses
    pos_subjects = [
        "The customer service team", "The latest app update", "The delivery driver",
        "The new battery life", "The user interface", "This wireless headset",
        "The onboarding tutorial", "The hotel concierge", "The engineering support",
        "The checkout process", "Our flight crew", "The camera optics",
        "The cloud backup system", "The sound quality", "The ergonomic keyboard",
        "The warranty replacement service", "The restaurant staff", "The analytics dashboard",
        "The software performance", "The payment gateway integration", "The mobile app"
    ]
    pos_actions = [
        "exceeded all my expectations", "worked like a charm on the first try",
        "made our workflow ten times faster", "resolved our complex issue in record time",
        "delivered outstanding results without any hassle", "provided exceptional and polite assistance",
        "feels remarkably premium and sturdy", "runs silky smooth with zero stutter",
        "made my day so much easier", "is worth every single cent invested",
        "blew me away with how intuitive it is", "saved our team countless hours this week",
        "is by far the best choice in this price category", "completely blew past our benchmark targets"
    ]
    pos_closings = [
        "Truly a 10/10 experience!", "Highly recommended to everyone.", "Kudos to the team!",
        "Couldn't be happier with this purchase.", "Will definitely order again.",
        "Keep up the brilliant work.", "Five stars all the way!", "Super impressed!"
    ]
    pos_standalone = [
        "Not bad at all, in fact it turned out to be one of the best tools I have used this year!",
        "I was slightly hesitant initially, but now I am completely in love with this product.",
        "Everything from the packaging to the build quality feels luxurious and thoughtful.",
        "The automated reporting feature works flawlessly and saves me hours each morning.",
        "Smooth, responsive, and beautifully designed user experience throughout.",
        "Customer support answered my ticket in less than two minutes and solved everything.",
        "The noise cancellation on these headphones makes working in noisy cafes a breeze.",
        "Stunning display clarity, vibrant colors, and very snappy refresh rate.",
        "Seamless synchronization between my laptop, tablet, and mobile device.",
        "Such a breath of fresh air compared to the bloated alternatives on the market.",
        "Delicious food, welcoming hospitality, and fantastic atmosphere overall.",
        "Genuinely thrilled with how fast the processing pipeline executes now.",
        "An absolute game changer for our engineering and marketing workflows.",
        "Reliable uptime and rock-solid stability over the last six months of production use.",
        "Simple setup, clear documentation, and great developer experience out of the box."
    ]

    # Varied Neutral phrases & clauses
    neu_subjects = [
        "The scheduled maintenance window", "The weekly team sync", "The billing department",
        "The delivery courier", "The software release version 2.4", "The user manual",
        "The database backup cron job", "Flight QR832", "The account verification link",
        "The quarterly financial report", "The office reception desk", "The return policy window",
        "The conference keynote session", "The package tracking identifier", "The subscription plan",
        "The server CPU load", "The API endpoint rate limit", "The train timetable",
        "The store operating hours", "The contract termination clause"
    ]
    neu_actions = [
        "is scheduled to take place between 02:00 and 04:00 UTC",
        "was updated to reflect standard operational guidelines",
        "requires a confirmation code sent via email",
        "will be hosted via Google Meet starting at 10:00 AM",
        "arrived at the regional sorting facility this morning",
        "contains detailed instructions in section four",
        "departs from terminal 2 at gate 14",
        "has been logged in the audit trail for review",
        "allows up to 100 requests per minute per IP address",
        "indicates an average response time of 240 milliseconds",
        "applies equally to monthly and annual tier subscribers",
        "was dispatched according to the standard postal service route",
        "remains valid for thirty calendar days from issue date",
        "is open from Monday through Friday between 9 AM and 6 PM"
    ]
    neu_closings = [
        "Please refer to the documentation for further details.",
        "No immediate action is required on your part.",
        "Contact the administration desk if you have any questions.",
        "Standard operating procedure applies as usual.",
        "Please retain this reference number for your records.",
        "Further updates will be posted on the status dashboard."
    ]
    neu_standalone = [
        "The package is currently in transit and expected to arrive on Thursday.",
        "Please note that the library will be closed during the upcoming national holiday.",
        "The system recorded an entry timestamp of 14:22:08 on the local gateway.",
        "You can modify your notification preferences under account privacy settings.",
        "The meeting minutes have been attached as a PDF file for your review.",
        "Train departures occur at thirty-minute intervals throughout the day.",
        "The software is compatible with Windows 11 and Ubuntu Linux 22.04 LTS.",
        "The temperature in the data center is currently maintained at 21 degrees Celsius.",
        "An automated confirmation has been forwarded to your registered email.",
        "The physical dimensions of the container measure 30 by 20 by 15 centimeters.",
        "Submissions must be received prior to midnight on the final day of the month.",
        "The workshop will cover fundamental concepts in data structures and algorithms.",
        "A copy of the formal receipt can be downloaded from the payment portal.",
        "The network switch operates at standard gigabit Ethernet transmission speeds.",
        "Please ensure all required input fields are populated before submitting the form."
    ]

    # Varied Negative phrases & clauses
    neg_subjects = [
        "The customer service department", "The recent mobile update", "The checkout payment system",
        "The product build quality", "The delivery carrier", "The battery performance",
        "The technical support agent", "The return and refund process", "The navigation menu",
        "The flight departure schedule", "The audio jack connection", "The server infrastructure",
        "The hardware cooling fan", "The web application interface", "The account security system",
        "The food quality", "The hotel room cleanliness", "The subscription billing mechanism"
    ]
    neg_actions = [
        "completely crashed and lost all of my unsaved progress",
        "refused to honor the warranty without providing any legitimate explanation",
        "is ridiculously slow, buggy, and frustrating to use",
        "broke apart into pieces after only two days of gentle use",
        "charged my credit card twice for a single order without authorization",
        "left me waiting on hold for forty-five minutes before disconnecting",
        "arrived severely damaged with pieces scattered inside the box",
        "drains the battery from 100 percent to zero in less than two hours",
        "is full of broken links and cryptic error messages",
        "was delayed by seven hours without food or accommodation vouchers",
        "overheats dangerously within five minutes of launching any task",
        "has completely broken our existing production workflow",
        "is completely unresponsive to critical support tickets",
        "feels cheap, flimsy, and not at all like advertised online"
    ]
    neg_closings = [
        "Never buying from this vendor again.",
        "Total waste of money and valuable time.",
        "Extremely disappointed with this experience.",
        "Demanding an immediate and full refund!",
        "Zero stars if the platform allowed it.",
        "Save your money and look for better alternatives.",
        "Utterly frustrating and unacceptable."
    ]
    neg_standalone = [
        "I regret buying this item; it is nowhere near as reliable as claimed.",
        "The app keeps freezing every time I try to upload a profile photo.",
        "Terrible experience from start to finish with unhelpful, rude representatives.",
        "The item arrived a week late, completely broken, and the seller ignores messages.",
        "Battery life is atrocious; it barely lasts through a single conference call.",
        "They charged me an unauthorized monthly renewal after I had already canceled.",
        "The food tasted stale and half of our dinner order was missing from the bag.",
        "Constant WiFi disconnection issues make this device practically useless.",
        "The instructions are contradictory and impossible to follow for beginners.",
        "Extremely noisy fan that whines incessantly even when the machine is idle.",
        "Misleading product description and deceptive pricing tactics. Avoid this company.",
        "The software update introduced massive lag and broke the export feature.",
        "Waited three weeks for a replacement unit only to receive another defective item.",
        "The display developed dead pixels across the center within a single week.",
        "Unprofessional conduct from the front desk staff ruined our entire stay."
    ]

    records = []

    # Generate Positive samples
    for i in range(samples_per_class):
        if i < len(pos_standalone):
            text = pos_standalone[i]
        else:
            subj = random.choice(pos_subjects)
            act = random.choice(pos_actions)
            close = random.choice(pos_closings)
            form = random.randint(0, 3)
            if form == 0:
                text = f"{subj} {act}. {close}"
            elif form == 1:
                text = f"{subj} {act}."
            elif form == 2:
                text = f"Honestly, {subj.lower()} {act}!"
            else:
                text = f"{close} {subj} {act}."
        records.append({"text": text, "sentiment": "positive"})

    # Generate Neutral samples
    for i in range(samples_per_class):
        if i < len(neu_standalone):
            text = neu_standalone[i]
        else:
            subj = random.choice(neu_subjects)
            act = random.choice(neu_actions)
            close = random.choice(neu_closings)
            form = random.randint(0, 3)
            if form == 0:
                text = f"{subj} {act}. {close}"
            elif form == 1:
                text = f"{subj} {act}."
            elif form == 2:
                text = f"Notice: {subj.lower()} {act}."
            else:
                text = f"{subj} {act}. ({close})"
        records.append({"text": text, "sentiment": "neutral"})

    # Generate Negative samples
    for i in range(samples_per_class):
        if i < len(neg_standalone):
            text = neg_standalone[i]
        else:
            subj = random.choice(neg_subjects)
            act = random.choice(neg_actions)
            close = random.choice(neg_closings)
            form = random.randint(0, 3)
            if form == 0:
                text = f"{subj} {act}. {close}"
            elif form == 1:
                text = f"{subj} {act}."
            elif form == 2:
                text = f"Very disappointed that {subj.lower()} {act}!"
            else:
                text = f"{close} {subj} {act}."
        records.append({"text": text, "sentiment": "negative"})

    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Dataset Created] Saved {len(df)} diverse samples to -> {output_path}")
    print(f"Class Distribution:\n{df['sentiment'].value_counts()}")
    return df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_csv = os.path.join(current_dir, "sentiment_data.csv")
    generate_curated_sentiment_dataset(target_csv, samples_per_class=400)
