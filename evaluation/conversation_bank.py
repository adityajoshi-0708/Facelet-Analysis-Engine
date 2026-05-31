"""
evaluation/conversation_bank.py — Phase 17A Step 2.

80 hand-authored benchmark conversations.
Each conversation covers at least one facet from Facets_Assignment.csv
and carries expected_facets + expected_scores for evaluation.

Usage:
    from evaluation.conversation_bank import load_conversations, CONVERSATIONS
    convs = load_conversations()          # list[dict]
    convs = load_conversations("hard")    # filter by difficulty
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Benchmark corpus
# Facet names match Facets_Assignment.csv spelling exactly.
# ---------------------------------------------------------------------------
CONVERSATIONS: List[dict] = [

    # ── PERSONALITY ──────────────────────────────────────────────────────────

    {
        "cid": "p01",
        "title": "Startup founder takes leap of faith",
        "category": "personality",
        "difficulty": "easy",
        "tags": ["risktaking", "determinedness", "adventure"],
        "turns": [
            {"speaker": "user", "text": "I quit my well-paying corporate job six months ago to start my own company. Everyone told me it was stupid, but I believed in the idea."},
            {"speaker": "user", "text": "I invested almost all my savings into it. No salary, no safety net. I love the uncertainty — it keeps me sharp."},
        ],
        "expected_facets": ["Risktaking", "Determinedness", "Adventure-Seeking Behavior"],
        "expected_scores": {"Risktaking": 5, "Determinedness": 4, "Adventure-Seeking Behavior": 4},
    },

    {
        "cid": "p02",
        "title": "Persistent athlete refuses to give up",
        "category": "personality",
        "difficulty": "easy",
        "tags": ["determinedness", "perseverance", "hardworking"],
        "turns": [
            {"speaker": "user", "text": "I've failed the qualifying exam three times. Most people would stop, but I registered again for next month."},
            {"speaker": "user", "text": "I train six hours a day. I genuinely believe I'll make it if I just keep going."},
        ],
        "expected_facets": ["Determinedness", "Perseverance", "Hardworking"],
        "expected_scores": {"Determinedness": 5, "Perseverance": 5, "Hardworking": 4},
    },

    {
        "cid": "p03",
        "title": "Deeply honest person admits a mistake publicly",
        "category": "personality",
        "difficulty": "easy",
        "tags": ["honesty", "genuine", "frankness"],
        "turns": [
            {"speaker": "user", "text": "I made a calculation error in the report I sent to 200 clients. I could have quietly fixed it, but I sent a correction email to everyone explaining exactly what I did wrong."},
            {"speaker": "user", "text": "I think people deserve the truth even when it's embarrassing for me."},
        ],
        "expected_facets": ["HonestyHumility:", "Genuine", "Frankness"],
        "expected_scores": {"HonestyHumility:": 5, "Genuine": 5, "Frankness": 4},
    },

    {
        "cid": "p04",
        "title": "Overly trusting person gets scammed repeatedly",
        "category": "personality",
        "difficulty": "medium",
        "tags": ["naivety", "trust"],
        "turns": [
            {"speaker": "user", "text": "This is the second time someone has taken advantage of me. I lent money to a new friend and he disappeared."},
            {"speaker": "user", "text": "I always assume people mean well. My therapist says I'm too trusting, but I just can't help it."},
        ],
        "expected_facets": ["Naivety", "Trust in others"],
        "expected_scores": {"Naivety": 5, "Trust in others": 4},
    },

    {
        "cid": "p05",
        "title": "Curious learner explores everything",
        "category": "personality",
        "difficulty": "easy",
        "tags": ["openness", "curiosity", "self-improvement"],
        "turns": [
            {"speaker": "user", "text": "I signed up for a course on quantum physics last week. I'm also reading about Byzantine history and learning Swahili."},
            {"speaker": "user", "text": "I genuinely love learning for its own sake — not for career reasons, just because it's fascinating."},
        ],
        "expected_facets": ["Openness", "Self-improvement", "Attitude toward learning"],
        "expected_scores": {"Openness": 5, "Self-improvement": 4, "Attitude toward learning": 5},
    },

    {
        "cid": "p06",
        "title": "Rebellious person refuses authority",
        "category": "personality",
        "difficulty": "medium",
        "tags": ["rebellious", "individuali", "nonconformist"],
        "turns": [
            {"speaker": "user", "text": "My boss gave a rule about dress code and I just ignored it. Rules that don't make sense shouldn't be followed."},
            {"speaker": "user", "text": "I've always pushed back on authority. I believe people should think for themselves, not just comply."},
        ],
        "expected_facets": ["Rebelliousness", "Individuality"],
        "expected_scores": {"Rebelliousness": 5, "Individuality": 4},
    },

    {
        "cid": "p07",
        "title": "Highly conscientious professional",
        "category": "personality",
        "difficulty": "easy",
        "tags": ["conscientiousness", "orderlines", "meeting deadlines"],
        "turns": [
            {"speaker": "user", "text": "I have never missed a deadline in eight years at this company. I keep color-coded folders and a weekly review system."},
            {"speaker": "user", "text": "I feel genuinely anxious if my workspace is cluttered. Everything has a place."},
        ],
        "expected_facets": ["Conscientiousness (C):", "Orderlines", "Meeting Deadlines"],
        "expected_scores": {"Conscientiousness (C):": 5, "Orderlines": 5, "Meeting Deadlines": 5},
    },

    # ── EMOTION ──────────────────────────────────────────────────────────────

    {
        "cid": "e01",
        "title": "Compassionate nurse describes her work",
        "category": "emotion",
        "difficulty": "easy",
        "tags": ["compassion", "empathy", "warmth"],
        "turns": [
            {"speaker": "user", "text": "I sat with a dying patient for three hours last night just so she wouldn't be alone. It wasn't in my job description."},
            {"speaker": "user", "text": "I carry their pain home with me. I can't just switch it off. I genuinely feel what they feel."},
        ],
        "expected_facets": ["Compassion", "Compassion Fatigue"],
        "expected_scores": {"Compassion": 5, "Compassion Fatigue": 4},
    },

    {
        "cid": "e02",
        "title": "Person overwhelmed by others' suffering",
        "category": "emotion",
        "difficulty": "medium",
        "tags": ["compassion fatigue", "emotionalism", "sensitivity"],
        "turns": [
            {"speaker": "user", "text": "I've been a social worker for twelve years and I think I'm burning out. I can't watch the news anymore."},
            {"speaker": "user", "text": "Every client's story stays with me. I cry in my car after difficult sessions. I know I need to build more distance but I don't know how."},
        ],
        "expected_facets": ["Compassion Fatigue", "Sensitiveness", "Emotionalism"],
        "expected_scores": {"Compassion Fatigue": 5, "Sensitiveness": 5, "Emotionalism": 4},
    },

    {
        "cid": "e03",
        "title": "Joyful person talks about daily happiness",
        "category": "emotion",
        "difficulty": "easy",
        "tags": ["joyfulness", "merriness", "contentment"],
        "turns": [
            {"speaker": "user", "text": "I just find so many things delightful. This morning a bird landed on my windowsill and it genuinely made my day."},
            {"speaker": "user", "text": "People tell me I'm relentlessly cheerful. I don't see the point in being otherwise — life is good."},
        ],
        "expected_facets": ["Joyfulness", "Merriness", "Contentment Levels"],
        "expected_scores": {"Joyfulness": 5, "Merriness": 5, "Contentment Levels": 5},
    },

    {
        "cid": "e04",
        "title": "Morose person reflects on persistent gloom",
        "category": "emotion",
        "difficulty": "medium",
        "tags": ["moroseness", "discontentment", "pessimism"],
        "turns": [
            {"speaker": "user", "text": "I've always had this grey cloud over me. Even on objectively good days I can't shake the feeling that something is wrong."},
            {"speaker": "user", "text": "I expect things to go wrong and they usually do. I'm not dramatic — that's just reality for me."},
        ],
        "expected_facets": ["Moroseness", "Discontentment"],
        "expected_scores": {"Moroseness": 5, "Discontentment": 4},
    },

    {
        "cid": "e05",
        "title": "Person with deep emotional boundaries",
        "category": "emotion",
        "difficulty": "medium",
        "tags": ["emotional boundaries", "aloofness", "withdrwanness"],
        "turns": [
            {"speaker": "user", "text": "I keep my emotions entirely separate from my professional life. I don't share personal things at work — ever."},
            {"speaker": "user", "text": "Some people find me cold. I prefer 'measured'. I don't think workplaces are the right place for feelings."},
        ],
        "expected_facets": ["Emotional Boundaries", "Aloofness", "Withrawnness"],
        "expected_scores": {"Emotional Boundaries": 5, "Aloofness": 4},
    },

    # ── COGNITION ────────────────────────────────────────────────────────────

    {
        "cid": "c01",
        "title": "Analyst describes rigorous statistical thinking",
        "category": "cognition",
        "difficulty": "easy",
        "tags": ["statistical reasoning", "critical reasoning", "analytical"],
        "turns": [
            {"speaker": "user", "text": "Before I accept any claim I ask about the sample size, the methodology, the confidence interval. Anecdotes don't move me."},
            {"speaker": "user", "text": "My team thinks I'm pedantic about data, but bad statistics cause real damage to decision-making."},
        ],
        "expected_facets": ["Statistical Reasoning", "Critical reasoning"],
        "expected_scores": {"Statistical Reasoning": 5, "Critical reasoning": 5},
    },

    {
        "cid": "c02",
        "title": "Person talks about poor common sense in daily life",
        "category": "cognition",
        "difficulty": "medium",
        "tags": ["common-sense", "naivety", "impracticalness"],
        "turns": [
            {"speaker": "user", "text": "I drove an hour to a store without checking if it was open. It was Sunday. I do this kind of thing constantly."},
            {"speaker": "user", "text": "My friends say I have zero common sense. I'm book-smart but practically useless."},
        ],
        "expected_facets": ["Common-sense", "Impracticalness"],
        "expected_scores": {"Common-sense": 1, "Impracticalness": 4},
    },

    {
        "cid": "c03",
        "title": "Fast-thinking problem solver under pressure",
        "category": "cognition",
        "difficulty": "easy",
        "tags": ["rapid cognitive processing", "decision-making speed"],
        "turns": [
            {"speaker": "user", "text": "In the emergency room you don't have time to deliberate. I make triage decisions in seconds and I'm right most of the time."},
            {"speaker": "user", "text": "Slow thinkers freeze. I process information and act — that's what saves lives."},
        ],
        "expected_facets": ["Rapid cognitive processing", "Decision-making speed", "Decision-Making Confidence"],
        "expected_scores": {"Rapid cognitive processing": 5, "Decision-making speed": 5, "Decision-Making Confidence": 5},
    },

    {
        "cid": "c04",
        "title": "Person who struggles with indecision",
        "category": "cognition",
        "difficulty": "medium",
        "tags": ["hesitation", "decisiveness", "anxiety"],
        "turns": [
            {"speaker": "user", "text": "I spent three weeks deciding which laptop to buy. I made a spreadsheet comparing 14 models and still wasn't sure."},
            {"speaker": "user", "text": "Every decision feels monumental to me. I'm terrified of making the wrong choice."},
        ],
        "expected_facets": ["Hesitation", "Decision-making decisiveness"],
        "expected_scores": {"Hesitation": 5, "Decision-making decisiveness": 1},
    },

    # ── RELATIONAL ───────────────────────────────────────────────────────────

    {
        "cid": "r01",
        "title": "Assertive negotiator in difficult meetings",
        "category": "relational",
        "difficulty": "easy",
        "tags": ["assertiveness", "social boldness", "outspokenness"],
        "turns": [
            {"speaker": "user", "text": "In yesterday's board meeting I told the CEO directly that his plan would fail. I laid out the reasons calmly and clearly."},
            {"speaker": "user", "text": "I'm not afraid to disagree with powerful people. If I see a problem I say so. That's just respect — for them and for the work."},
        ],
        "expected_facets": ["Assertiveness and control in relationships", "Social Boldness", "Outspokenness"],
        "expected_scores": {"Assertiveness and control in relationships": 5, "Social Boldness": 5, "Outspokenness": 5},
    },

    {
        "cid": "r02",
        "title": "Passive-aggressive colleague",
        "category": "relational",
        "difficulty": "hard",
        "tags": ["passive-aggressive", "hostility", "indirect"],
        "turns": [
            {"speaker": "user", "text": "I didn't say anything when my manager took credit for my work. I just started making sure all my emails were CC'd to her boss."},
            {"speaker": "user", "text": "I never confront people directly. I find ways to make my point without having the actual argument."},
        ],
        "expected_facets": ["Passive-Aggressive", "Hostility"],
        "expected_scores": {"Passive-Aggressive": 5, "Hostility": 3},
    },

    {
        "cid": "r03",
        "title": "Deeply collaborative team member",
        "category": "relational",
        "difficulty": "easy",
        "tags": ["collaboration", "cooperation", "relationship building"],
        "turns": [
            {"speaker": "user", "text": "I deliberately involve the quietest voices in every meeting. Good ideas come from everywhere."},
            {"speaker": "user", "text": "I share credit aggressively. If we succeed, it's because of everyone, and I make sure they know they matter."},
        ],
        "expected_facets": ["Collaboration", "Cooperation", "Relationship Building Themes:"],
        "expected_scores": {"Collaboration": 5, "Cooperation": 5},
    },

    {
        "cid": "r04",
        "title": "Suspicious person trusts no one",
        "category": "relational",
        "difficulty": "medium",
        "tags": ["suspicion", "trust", "hostility"],
        "turns": [
            {"speaker": "user", "text": "I assume people have ulterior motives until proven otherwise. At work, at home — everywhere."},
            {"speaker": "user", "text": "My ex betrayed me badly and since then I keep everyone at arm's length. I check everything twice."},
        ],
        "expected_facets": ["Suspicion", "Trust in others"],
        "expected_scores": {"Suspicion": 5, "Trust in others": 1},
    },

    {
        "cid": "r05",
        "title": "Submissive person can't say no",
        "category": "relational",
        "difficulty": "medium",
        "tags": ["submissiveness", "unassertiveness", "seeking approval"],
        "turns": [
            {"speaker": "user", "text": "I agreed to work over Christmas even though I had plans. I just couldn't say no to my boss."},
            {"speaker": "user", "text": "I need people to like me. I end up doing things I resent because I can't stand the idea of disappointing anyone."},
        ],
        "expected_facets": ["Submissiveness", "Unassertiveness", "Seeking approval"],
        "expected_scores": {"Submissiveness": 5, "Unassertiveness": 5, "Seeking approval": 5},
    },

    {
        "cid": "r06",
        "title": "Caring but overprotective parent",
        "category": "relational",
        "difficulty": "medium",
        "tags": ["overprotectiveness", "affection", "control"],
        "turns": [
            {"speaker": "user", "text": "My daughter is 19 and I still track her phone. I know it's a lot but I just can't stop worrying."},
            {"speaker": "user", "text": "I love her more than anything. I just can't bear the thought of something happening to her."},
        ],
        "expected_facets": ["Overprotectiveness", "Affection"],
        "expected_scores": {"Overprotectiveness": 5, "Affection": 5},
    },

    # ── LEADERSHIP ───────────────────────────────────────────────────────────

    {
        "cid": "l01",
        "title": "Democratic leader who values all voices",
        "category": "leadership",
        "difficulty": "easy",
        "tags": ["democratic leadership", "collaboration", "delegation"],
        "turns": [
            {"speaker": "user", "text": "Before any major decision I hold a team forum. I don't move forward unless I understand every perspective, even if it slows us down."},
            {"speaker": "user", "text": "My job as a leader is to create the conditions for other people to be brilliant, not to be the smartest person in the room."},
        ],
        "expected_facets": ["Democratic Leadership:", "Delegation Ability", "Encouraging participation"],
        "expected_scores": {"Democratic Leadership:": 5, "Delegation Ability": 4, "Encouraging participation": 5},
    },

    {
        "cid": "l02",
        "title": "Controlling manager who micromanages everything",
        "category": "leadership",
        "difficulty": "medium",
        "tags": ["control", "transactional leadership", "structure"],
        "turns": [
            {"speaker": "user", "text": "I review every piece of work before it goes out. My team thinks it's annoying but errors have consequences."},
            {"speaker": "user", "text": "I don't delegate easily. By the time I've explained it, I could have done it myself."},
        ],
        "expected_facets": ["Transactional Leadership:", "Structure", "Control over situations"],
        "expected_scores": {"Transactional Leadership:": 4, "Structure": 5, "Control over situations": 5},
    },

    {
        "cid": "l03",
        "title": "Leader who invests in developing others",
        "category": "leadership",
        "difficulty": "easy",
        "tags": ["leadership potential", "delegation", "mentoring"],
        "turns": [
            {"speaker": "user", "text": "I spend 20% of my week doing one-on-ones focused purely on my team's growth. Not project updates — their careers."},
            {"speaker": "user", "text": "Three people I've mentored have become leaders themselves. That's the real measure of leadership."},
        ],
        "expected_facets": ["Leadership Potential:", "Delegation skills", "Desire to influence others"],
        "expected_scores": {"Leadership Potential:": 5, "Delegation skills": 4, "Desire to influence others": 4},
    },

    # ── BEHAVIORAL ───────────────────────────────────────────────────────────

    {
        "cid": "b01",
        "title": "Compulsive shopper describes spending habits",
        "category": "behavioral",
        "difficulty": "medium",
        "tags": ["compulsive", "impulsivity", "self-control"],
        "turns": [
            {"speaker": "user", "text": "I bought four jackets this month and I don't need any of them. Once I see something I want, I just buy it immediately."},
            {"speaker": "user", "text": "My bank account is suffering. I know I need to stop but the urge is overwhelming in the moment."},
        ],
        "expected_facets": ["Compulsive activities", "Impulsivity", "Selfcontrol"],
        "expected_scores": {"Compulsive activities": 5, "Impulsivity": 5, "Selfcontrol": 1},
    },

    {
        "cid": "b02",
        "title": "Highly self-disciplined morning routine",
        "category": "behavioral",
        "difficulty": "easy",
        "tags": ["self-control", "discipline", "organized"],
        "turns": [
            {"speaker": "user", "text": "5 AM every day: 30-minute run, cold shower, journaling, then two hours of deep work before the kids wake up."},
            {"speaker": "user", "text": "I've maintained this routine for four years. Discipline is just decisions made in advance."},
        ],
        "expected_facets": ["Selfcontrol", "Organized lifestyle", "Hardworking"],
        "expected_scores": {"Selfcontrol": 5, "Organized lifestyle": 5, "Hardworking": 5},
    },

    {
        "cid": "b03",
        "title": "Adventurer who seeks novelty constantly",
        "category": "behavioral",
        "difficulty": "easy",
        "tags": ["adventure seeking", "boredom susceptibility", "risktaking"],
        "turns": [
            {"speaker": "user", "text": "I've visited 47 countries in five years. I get genuinely restless if I'm in the same place for more than three weeks."},
            {"speaker": "user", "text": "Routine kills me. I need novelty — new food, new culture, new challenge. Safety is overrated."},
        ],
        "expected_facets": ["Adventure-Seeking Behavior", "Boredom Susceptibility", "Risktaking"],
        "expected_scores": {"Adventure-Seeking Behavior": 5, "Boredom Susceptibility": 5, "Risktaking": 4},
    },

    {
        "cid": "b04",
        "title": "Volunteer shares motivation for community work",
        "category": "behavioral",
        "difficulty": "easy",
        "tags": ["volunteer", "big-heartedness", "significance"],
        "turns": [
            {"speaker": "user", "text": "I run a food drive every month. It started as a one-off but now I can't imagine not doing it."},
            {"speaker": "user", "text": "Giving time feels more meaningful than giving money. I want to make a tangible difference."},
        ],
        "expected_facets": ["Volunteer Work", "Big-heartedness", "Significance: Desire to make an impact"],
        "expected_scores": {"Volunteer Work": 5, "Big-heartedness": 5, "Significance: Desire to make an impact": 4},
    },

    {
        "cid": "b05",
        "title": "Slothful person avoids all effort",
        "category": "behavioral",
        "difficulty": "easy",
        "tags": ["slothfulness", "inefficiency", "procrastination"],
        "turns": [
            {"speaker": "user", "text": "I had three months to write a 2000-word report. I started at 11pm the night before."},
            {"speaker": "user", "text": "I don't see the point in doing things before I absolutely have to. It usually works out fine."},
        ],
        "expected_facets": ["Slothfulness", "Inefficiency"],
        "expected_scores": {"Slothfulness": 5, "Inefficiency": 4},
    },

    # ── MENTAL HEALTH ────────────────────────────────────────────────────────

    {
        "cid": "mh01",
        "title": "Person describes ongoing depression symptoms",
        "category": "clinical",
        "difficulty": "hard",
        "tags": ["depression", "moroseness", "withdrawal"],
        "turns": [
            {"speaker": "user", "text": "I haven't gotten out of bed before noon in two weeks. I just lie there staring at the ceiling."},
            {"speaker": "user", "text": "Nothing feels worth doing. I used to love cooking and I can't even make toast right now."},
        ],
        "expected_facets": ["Depression Symptoms", "Depression: Feelings of sadness and hopelessness"],
        "expected_scores": {"Depression Symptoms": 5, "Depression: Feelings of sadness and hopelessness": 5},
    },

    {
        "cid": "mh02",
        "title": "Burnout after years of overwork",
        "category": "clinical",
        "difficulty": "medium",
        "tags": ["burnout", "compassion fatigue", "stress"],
        "turns": [
            {"speaker": "user", "text": "I used to care deeply about my work. Now I show up and go through the motions. I feel nothing."},
            {"speaker": "user", "text": "My doctor said I'm burned out. I didn't even argue. I've had nothing left for months."},
        ],
        "expected_facets": ["Burnout Symptoms"],
        "expected_scores": {"Burnout Symptoms": 5},
    },

    {
        "cid": "mh03",
        "title": "Person with chronic stress and irritability",
        "category": "clinical",
        "difficulty": "medium",
        "tags": ["irritability", "stress", "negative affect"],
        "turns": [
            {"speaker": "user", "text": "I snapped at my kid for leaving a cup on the counter. That's not me — I've been on edge for months."},
            {"speaker": "user", "text": "Small things set me off now. I'm exhausted all the time and nothing feels manageable."},
        ],
        "expected_facets": ["Irritability", "Negative Affect Frequency", "Stress Recovery Ability"],
        "expected_scores": {"Irritability": 5, "Negative Affect Frequency": 5, "Stress Recovery Ability": 1},
    },

    # ── COGNITION / SELF ─────────────────────────────────────────────────────

    {
        "cid": "cs01",
        "title": "Person with low self-esteem shares insecurities",
        "category": "cognition",
        "difficulty": "hard",
        "tags": ["self-esteem", "self-efficacy", "insecurity"],
        "turns": [
            {"speaker": "user", "text": "I got promoted but I'm convinced it's a mistake. Sooner or later they'll realise I'm not good enough."},
            {"speaker": "user", "text": "I prepare twice as hard as anyone else because I'm terrified of being found out."},
        ],
        "expected_facets": ["SelfEsteem", "Self-Efficacy"],
        "expected_scores": {"SelfEsteem": 1, "Self-Efficacy": 2},
    },

    {
        "cid": "cs02",
        "title": "Self-aware person reflects honestly",
        "category": "cognition",
        "difficulty": "medium",
        "tags": ["self-perspective", "self-reflection", "genuine"],
        "turns": [
            {"speaker": "user", "text": "I know I tend to be too blunt. I've hurt people with honesty that wasn't asked for. I'm working on that."},
            {"speaker": "user", "text": "Self-awareness doesn't fix the problem automatically, but at least I can see what I'm doing."},
        ],
        "expected_facets": ["Self Perspective", "Regularity of Self-Reflection"],
        "expected_scores": {"Self Perspective": 4, "Regularity of Self-Reflection": 5},
    },

    {
        "cid": "cs03",
        "title": "Highly self-righteous person judges others",
        "category": "cognition",
        "difficulty": "hard",
        "tags": ["self-righteousness", "judging", "moralism"],
        "turns": [
            {"speaker": "user", "text": "I don't understand how people can eat meat when the evidence about its harm is so clear. It's just laziness and selfishness."},
            {"speaker": "user", "text": "I hold myself to a very high standard and I expect the same from others. Most people don't try hard enough."},
        ],
        "expected_facets": ["Self-righteousness", "Judging (J):"],
        "expected_scores": {"Self-righteousness": 5, "Judging (J):": 4},
    },

    # ── SPIRITUALITY ─────────────────────────────────────────────────────────

    {
        "cid": "sp01",
        "title": "Person describes their spiritual practice",
        "category": "spirituality",
        "difficulty": "medium",
        "tags": ["spirituality", "mindfulness", "holiness"],
        "turns": [
            {"speaker": "user", "text": "I meditate for an hour every morning. Not as a wellness trend — it's how I connect with something larger than myself."},
            {"speaker": "user", "text": "My faith shapes every decision I make. I try to live according to principles of compassion and non-harm."},
        ],
        "expected_facets": ["Presence of Spiritual Pain", "Mindfulness facet: Observing", "Role of Spirituality in Community Involvement"],
        "expected_scores": {"Mindfulness facet: Observing": 5},
    },

    {
        "cid": "sp02",
        "title": "Person experiencing spiritual pain",
        "category": "spirituality",
        "difficulty": "hard",
        "tags": ["spiritual pain", "existential", "meaning"],
        "turns": [
            {"speaker": "user", "text": "After my son died I couldn't pray anymore. Everything I believed felt hollow."},
            {"speaker": "user", "text": "The people who say 'he's in a better place' make it worse. I feel abandoned by everything I used to find meaning in."},
        ],
        "expected_facets": ["Presence of Spiritual Pain"],
        "expected_scores": {"Presence of Spiritual Pain": 5},
    },

    # ── MIXED ────────────────────────────────────────────────────────────────

    {
        "cid": "mx01",
        "title": "Complex entrepreneur: risk + compassion + leadership",
        "category": "mixed",
        "difficulty": "hard",
        "tags": ["risktaking", "compassion", "leadership", "determinedness"],
        "turns": [
            {"speaker": "user", "text": "I started a social enterprise in a conflict zone. Everyone said I was crazy."},
            {"speaker": "user", "text": "I pay my employees 20% above market and gave equity to the cleaners. Profit without humanity isn't worth it."},
            {"speaker": "user", "text": "The business nearly failed twice. I remortgaged my house both times. I couldn't let those families down."},
        ],
        "expected_facets": ["Risktaking", "Compassion", "Determinedness", "Democratic Leadership:"],
        "expected_scores": {"Risktaking": 5, "Compassion": 5, "Determinedness": 5},
    },

    {
        "cid": "mx02",
        "title": "Anxious achiever: ambition and self-doubt",
        "category": "mixed",
        "difficulty": "hard",
        "tags": ["achievement motivation", "self-esteem", "perfectionism"],
        "turns": [
            {"speaker": "user", "text": "I graduated top of my class, got a competitive job, and feel like a complete fraud every single day."},
            {"speaker": "user", "text": "I keep achieving things hoping it'll fill the void, but it never does. I'm exhausted from performing competence I don't feel."},
        ],
        "expected_facets": ["Achievement Motivation:", "SelfEsteem", "Psychological construct: Perfectionistic Strivings"],
        "expected_scores": {"Achievement Motivation:": 5, "SelfEsteem": 1},
    },

    {
        "cid": "mx03",
        "title": "Burned-out caregiver with deep empathy",
        "category": "mixed",
        "difficulty": "hard",
        "tags": ["compassion fatigue", "empathy", "burnout"],
        "turns": [
            {"speaker": "user", "text": "I've been caring for my mother with dementia for three years. I love her deeply but I'm disappearing."},
            {"speaker": "user", "text": "I feel her confusion and fear as if it were my own. That's beautiful and it's destroying me at the same time."},
        ],
        "expected_facets": ["Compassion Fatigue", "Burnout Symptoms"],
        "expected_scores": {"Compassion Fatigue": 5, "Burnout Symptoms": 4},
    },

    # ── AMBIGUOUS ────────────────────────────────────────────────────────────

    {
        "cid": "am01",
        "title": "Friend is selfish — attribution ambiguity",
        "category": "ambiguous",
        "difficulty": "hard",
        "tags": ["attribution", "disrespect", "mentioned entity"],
        "turns": [
            {"speaker": "user", "text": "My friend is incredibly selfish. He never shows up when I need him but expects me to drop everything for him."},
            {"speaker": "user", "text": "I keep letting him do this. I don't know why I put up with it."},
        ],
        "expected_facets": ["Unassertiveness", "Seeking approval"],
        "expected_scores": {"Unassertiveness": 4, "Seeking approval": 3},
        "notes": "Disrespect/selfishness belongs to mentioned friend, NOT the speaker.",
    },

    {
        "cid": "am02",
        "title": "Minimal-signal conversation",
        "category": "ambiguous",
        "difficulty": "hard",
        "tags": ["low signal", "ambiguous"],
        "turns": [
            {"speaker": "user", "text": "I had a pretty normal week. Went to work, came home, watched some TV."},
        ],
        "expected_facets": [],
        "expected_scores": {},
        "notes": "Very low signal — retriever should return low-confidence results.",
    },

    # ── ADDITIONAL PERSONALITY ───────────────────────────────────────────────

    {
        "cid": "p08",
        "title": "Cunning strategist describes manipulation",
        "category": "personality",
        "difficulty": "hard",
        "tags": ["cunningness", "dishonesty", "strategist"],
        "turns": [
            {"speaker": "user", "text": "I let people think my plan was their idea. They're more committed that way. It's just psychology."},
            {"speaker": "user", "text": "I'm not dishonest — I'm strategic. There's a difference between lying and just not revealing everything."},
        ],
        "expected_facets": ["Cunningness", "Dishonesty"],
        "expected_scores": {"Cunningness": 5, "Dishonesty": 3},
    },

    {
        "cid": "p09",
        "title": "Highly ethical professional describes principles",
        "category": "personality",
        "difficulty": "easy",
        "tags": ["ethical standards", "integrity", "honesty"],
        "turns": [
            {"speaker": "user", "text": "I walked away from a contract worth £200k because the client wanted me to misrepresent data. Not even close to a hard decision."},
            {"speaker": "user", "text": "My reputation for integrity is the most valuable thing I have. I won't trade it for any amount."},
        ],
        "expected_facets": ["Ethical standards", "HonestyHumility:", "Genuine"],
        "expected_scores": {"Ethical standards": 5, "HonestyHumility:": 5},
    },

    {
        "cid": "p10",
        "title": "Acidity in interactions — sharp and abrasive",
        "category": "personality",
        "difficulty": "medium",
        "tags": ["acidity", "cantankerousness", "coarseness"],
        "turns": [
            {"speaker": "user", "text": "People keep telling me my feedback is 'too harsh'. I think they want to be coddled."},
            {"speaker": "user", "text": "I don't see the point in softening things. If something is bad, I say it's bad. Sensitivity is overrated."},
        ],
        "expected_facets": ["Acidity", "Cantankerousness", "Coarseness"],
        "expected_scores": {"Acidity": 5, "Cantankerousness": 4, "Coarseness": 3},
    },

    {
        "cid": "p11",
        "title": "Chivalrous person describes old-fashioned values",
        "category": "personality",
        "difficulty": "medium",
        "tags": ["chivalrousness", "dignity", "decency"],
        "turns": [
            {"speaker": "user", "text": "I always hold doors, give up my seat, help carry bags — not because I have to but because it's the right thing to do."},
            {"speaker": "user", "text": "People mock it as old-fashioned. I think basic courtesy is timeless."},
        ],
        "expected_facets": ["Chivalrousness", "Dignity", "Decency"],
        "expected_scores": {"Chivalrousness": 5, "Dignity": 4, "Decency": 5},
    },

    {
        "cid": "p12",
        "title": "Genuine person who rejects performance",
        "category": "personality",
        "difficulty": "easy",
        "tags": ["genuine", "authenticity", "individuality"],
        "turns": [
            {"speaker": "user", "text": "I don't perform happiness or professionalism. What you see is what you get, at work and everywhere else."},
            {"speaker": "user", "text": "I've lost jobs for being too direct about my opinions. I'm fine with that. Playing a role is exhausting."},
        ],
        "expected_facets": ["Genuine", "Individuality"],
        "expected_scores": {"Genuine": 5, "Individuality": 4},
    },

    # ── MORE RELATIONAL ──────────────────────────────────────────────────────

    {
        "cid": "r07",
        "title": "Talker who dominates every conversation",
        "category": "relational",
        "difficulty": "medium",
        "tags": ["talkativeness", "social boldness", "outspokenness"],
        "turns": [
            {"speaker": "user", "text": "I talk a lot — I know it. I fill silences automatically. I'm thinking out loud most of the time."},
            {"speaker": "user", "text": "My partner says I need to learn to listen. She's probably right. But I have a lot to say."},
        ],
        "expected_facets": ["Talkativeness", "Outspokenness"],
        "expected_scores": {"Talkativeness": 5, "Outspokenness": 4},
    },

    {
        "cid": "r08",
        "title": "Conflict-averse person always compromises",
        "category": "relational",
        "difficulty": "medium",
        "tags": ["compromise", "conflict avoidance", "peacefulness"],
        "turns": [
            {"speaker": "user", "text": "I hate conflict so much I'll give up things I want to avoid an argument. It's not healthy but it's how I'm wired."},
            {"speaker": "user", "text": "My default is to find the middle ground. Even when I'm right I soften my position."},
        ],
        "expected_facets": ["Tendency Toward Compromise or Confrontation", "Peacefulness"],
        "expected_scores": {"Tendency Toward Compromise or Confrontation": 5, "Peacefulness": 4},
    },

    {
        "cid": "r09",
        "title": "Social person describes need for others",
        "category": "relational",
        "difficulty": "easy",
        "tags": ["social interaction", "affiliation motivation", "warmth"],
        "turns": [
            {"speaker": "user", "text": "I recharge by being around people. Solitude for more than a day makes me genuinely anxious."},
            {"speaker": "user", "text": "I know all my neighbours, most of the shop owners on my street, the regulars at my gym. Community is everything to me."},
        ],
        "expected_facets": ["Need for social interaction", "Affiliation Motivation:", "Warmheartedness"],
        "expected_scores": {"Need for social interaction": 5, "Affiliation Motivation:": 5, "Warmheartedness": 4},
    },

    # ── MORE COGNITION ───────────────────────────────────────────────────────

    {
        "cid": "c05",
        "title": "Creative problem solver describes novel approach",
        "category": "cognition",
        "difficulty": "easy",
        "tags": ["creativity", "innovation", "originality"],
        "turns": [
            {"speaker": "user", "text": "Everyone was stuck on the same approach for six months. I suggested solving it backwards — from the end state — and we cracked it in a week."},
            {"speaker": "user", "text": "I naturally see things from unusual angles. I get bored when everyone's converging on the obvious answer."},
        ],
        "expected_facets": ["Innovation and Creativity Components:", "Creativity in solutions", "Originality"],
        "expected_scores": {"Innovation and Creativity Components:": 5, "Creativity in solutions": 5, "Originality": 5},
    },

    {
        "cid": "c06",
        "title": "Person who struggles to synthesise information",
        "category": "cognition",
        "difficulty": "medium",
        "tags": ["synthesis", "information retention", "learning"],
        "turns": [
            {"speaker": "user", "text": "I read a lot but I struggle to connect the ideas. I'll read three books on the same topic and still can't say what the throughline is."},
            {"speaker": "user", "text": "My notes are detailed but disorganised. I absorb facts but miss the bigger picture."},
        ],
        "expected_facets": ["Synthesis of information", "Information Retention"],
        "expected_scores": {"Synthesis of information": 1, "Information Retention": 3},
    },

    # ── MORE BEHAVIORAL ──────────────────────────────────────────────────────

    {
        "cid": "b06",
        "title": "Person with strong goal-directed behavior",
        "category": "behavioral",
        "difficulty": "easy",
        "tags": ["goal directed", "achievement motivation", "initiative"],
        "turns": [
            {"speaker": "user", "text": "I have a five-year plan broken into quarterly milestones. Every Sunday I review progress and adjust."},
            {"speaker": "user", "text": "I don't wait to be asked. If I see something that needs doing I do it — and then tell people afterward."},
        ],
        "expected_facets": ["Goal-Directed Behavior", "Achievement Motivation:", "Initiative"],
        "expected_scores": {"Goal-Directed Behavior": 5, "Achievement Motivation:": 5, "Initiative": 5},
    },

    {
        "cid": "b07",
        "title": "Person describes deep conformity pressure",
        "category": "behavioral",
        "difficulty": "medium",
        "tags": ["conformity", "social conformity", "alignment"],
        "turns": [
            {"speaker": "user", "text": "I dress differently when I go home to my parents' community. It's just easier than the judgment."},
            {"speaker": "user", "text": "I've spent most of my life fitting in. I'm not sure I even know what I actually prefer anymore."},
        ],
        "expected_facets": ["Conformity", "Alignment with societal roles"],
        "expected_scores": {"Conformity": 5, "Alignment with societal roles": 5},
    },

    # ── MORE MIXED ───────────────────────────────────────────────────────────

    {
        "cid": "mx04",
        "title": "Transparent leader with high emotional intelligence",
        "category": "mixed",
        "difficulty": "medium",
        "tags": ["emotional intelligence", "genuine", "leadership"],
        "turns": [
            {"speaker": "user", "text": "I told my team last week that I was struggling with the direction the company was taking. I didn't pretend I had all the answers."},
            {"speaker": "user", "text": "Vulnerability from a leader isn't weakness. It models the psychological safety the team needs to speak up."},
        ],
        "expected_facets": ["Genuine", "Psychological construct: Psychological Safety climate", "Comfort with Vulnerability"],
        "expected_scores": {"Genuine": 5, "Comfort with Vulnerability": 5},
    },

    {
        "cid": "mx05",
        "title": "Curious but disorganised creative",
        "category": "mixed",
        "difficulty": "medium",
        "tags": ["curiosity", "openness", "disorganised"],
        "turns": [
            {"speaker": "user", "text": "I'm simultaneously writing a novel, learning woodworking, and teaching myself Japanese. I have seventeen browser tabs open about Byzantine art."},
            {"speaker": "user", "text": "I finish almost nothing but I know a little about everything. My desk looks like a minor explosion."},
        ],
        "expected_facets": ["Openness", "Boredom Susceptibility"],
        "expected_scores": {"Openness": 5, "Boredom Susceptibility": 4},
    },

    {
        "cid": "mx06",
        "title": "Disrespectful but high-achieving person",
        "category": "mixed",
        "difficulty": "hard",
        "tags": ["disrespect", "achievement motivation", "hostile"],
        "turns": [
            {"speaker": "user", "text": "I don't have patience for people who haven't done the work. I'll interrupt, I'll redirect, I'll move on. Time is too valuable."},
            {"speaker": "user", "text": "My results speak for themselves. People can be uncomfortable with my style if they want. I'm not here to make friends."},
        ],
        "expected_facets": ["Disrespect", "Achievement Motivation:"],
        "expected_scores": {"Disrespect": 4, "Achievement Motivation:": 5},
    },

    # ── WELLBEING ────────────────────────────────────────────────────────────

    {
        "cid": "w01",
        "title": "Person describes deep sense of purpose",
        "category": "behavioral",
        "difficulty": "easy",
        "tags": ["significance", "well-being", "engagement"],
        "turns": [
            {"speaker": "user", "text": "I wake up excited. My work feels genuinely important — I'm not just earning a salary."},
            {"speaker": "user", "text": "Even on hard days there's this underlying sense that what I'm doing matters. I feel connected to something bigger than myself."},
        ],
        "expected_facets": ["Significance: Desire to make an impact", "Well-being component: Engagement", "Intrinsic Motivation"],
        "expected_scores": {"Significance: Desire to make an impact": 5, "Well-being component: Engagement": 5, "Intrinsic Motivation": 5},
    },

    {
        "cid": "w02",
        "title": "Person struggling with comfort zone",
        "category": "behavioral",
        "difficulty": "medium",
        "tags": ["comfort with vulnerability", "adaptability", "stress"],
        "turns": [
            {"speaker": "user", "text": "I avoid situations where I might not know the answer. Public speaking, new social situations — I steer clear."},
            {"speaker": "user", "text": "I know I'm limiting myself. I just can't deal with the exposure of not being competent."},
        ],
        "expected_facets": ["Comfort with Vulnerability", "Adaptability and Flexibility:"],
        "expected_scores": {"Comfort with Vulnerability": 1, "Adaptability and Flexibility:": 1},
    },

    {
        "cid": "w03",
        "title": "Highly patriotic person describes national identity",
        "category": "personality",
        "difficulty": "medium",
        "tags": ["patriotism", "cultural identity", "conformity"],
        "turns": [
            {"speaker": "user", "text": "My country isn't perfect but I'm genuinely proud to be from here. I'd defend it without hesitation."},
            {"speaker": "user", "text": "I think people who denigrate their own country without offering anything better are just fashionably cynical."},
        ],
        "expected_facets": ["Patriotism", "Cultural Identity"],
        "expected_scores": {"Patriotism": 5, "Cultural Identity": 4},
    },

    {
        "cid": "w04",
        "title": "Person with strong justice orientation",
        "category": "personality",
        "difficulty": "easy",
        "tags": ["justice mindedness", "ethical standards", "morality"],
        "turns": [
            {"speaker": "user", "text": "I reported my own manager for falsifying expenses. I knew it would be awkward. Couldn't stay silent."},
            {"speaker": "user", "text": "Rules exist for a reason. When powerful people break them and nothing happens, it damages everyone."},
        ],
        "expected_facets": ["Justice-mindedness", "Ethical standards", "Moral and Ethical Parameters:"],
        "expected_scores": {"Justice-mindedness": 5, "Ethical standards": 5},
    },

    {
        "cid": "w05",
        "title": "Multiculturalist describes their worldview",
        "category": "personality",
        "difficulty": "easy",
        "tags": ["multiculturalism", "openness", "value universalism"],
        "turns": [
            {"speaker": "user", "text": "I've lived in six countries and I genuinely think every culture has something important to teach. There's no hierarchy."},
            {"speaker": "user", "text": "When I'm in a new place I try to experience it on its own terms rather than comparing it to home."},
        ],
        "expected_facets": ["Multiculturalism", "Openness", "Value orientation: Universalism"],
        "expected_scores": {"Multiculturalism": 5, "Openness": 5},
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_conversations(
    difficulty: Optional[str] = None,
    category: Optional[str] = None,
) -> List[dict]:
    """Return benchmark conversations, optionally filtered."""
    result = CONVERSATIONS
    if difficulty:
        result = [c for c in result if c.get("difficulty") == difficulty]
    if category:
        result = [c for c in result if c.get("category") == category]
    return result


def save_bank(path: Optional[Path] = None) -> Path:
    """Persist the conversation bank to JSON."""
    out = path or (Path(__file__).resolve().parents[1] / "evaluation" / "conversation_bank.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(CONVERSATIONS, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = save_bank()
    print(f"✓ Saved {len(CONVERSATIONS)} conversations → {p}")
    cats = {}
    for c in CONVERSATIONS:
        cats[c["category"]] = cats.get(c["category"], 0) + 1
    for cat, n in sorted(cats.items()):
        print(f"  {cat:20s}: {n}")