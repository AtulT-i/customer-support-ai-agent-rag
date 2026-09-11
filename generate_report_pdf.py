import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Running Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(36, 11 * inch - 28, "Hiver SDE Intern Take-Home — AI Support Agent Technical Report")
            self.drawRightString(8.5 * inch - 36, 11 * inch - 28, "Atul Kumar Tiwari | JIIT")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 11 * inch - 32, 8.5 * inch - 36, 11 * inch - 32)
            
        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 30, 8.5 * inch - 36, 30)
        
        self.drawString(36, 18, "Confidential — Prepared for Hiver SDE Internship Assessment (anurag@hiverhq.com)")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 36, 18, page_str)
        self.restoreState()

def build_pdf(filename="Hiver_SDE_Intern_Report_Atul_Kumar_Tiwari.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0F172A")    # Slate 900
    c_brand = colors.HexColor("#1DB954")      # Spotify Green
    c_accent = colors.HexColor("#2563EB")     # Royal Blue
    c_dark = colors.HexColor("#1E293B")       # Slate 800
    c_body = colors.HexColor("#334155")       # Slate 700
    c_muted = colors.HexColor("#64748B")      # Slate 500
    c_bg_light = colors.HexColor("#F8FAFC")   # Slate 50
    c_border = colors.HexColor("#CBD5E1")     # Slate 300

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=17, leading=21,
        textColor=c_primary, spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9.5, leading=13,
        textColor=c_accent, spaceAfter=7
    )
    meta_style = ParagraphStyle(
        'MetaText', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.8, leading=10.5,
        textColor=c_body
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=11, leading=14,
        textColor=c_primary, spaceBefore=6, spaceAfter=3,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8.8, leading=11.5,
        textColor=c_dark, spaceBefore=4, spaceAfter=2,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.8, leading=10.4,
        textColor=c_body, spaceAfter=3
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.8, leading=10.2,
        textColor=c_body, leftIndent=9, firstLineIndent=-6, spaceAfter=1.5
    )
    callout_style = ParagraphStyle(
        'CalloutText', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=7.6, leading=10.2,
        textColor=c_dark
    )
    table_cell = ParagraphStyle(
        'TableCell', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.2, leading=9.0,
        textColor=c_body
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=7.2, leading=9.0,
        textColor=c_primary
    )
    table_hdr = ParagraphStyle(
        'TableHdr', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=7.4, leading=9.2,
        textColor=colors.white
    )

    story = []

    # ==========================================
    # PAGE 1: HEADER, METADATA, PROBLEM & SCOPE
    # ==========================================
    story.append(Paragraph("Autonomous Customer Support AI Agent with Guardrailed RAG", title_style))
    story.append(Paragraph("A Production-Grade Triage, Extractive Retrieval & Deterministic Escalation System Built on Twitter Data", subtitle_style))
    
    # Metadata Block Table
    meta_data = [
        [
            Paragraph("<b>Candidate:</b> Atul Kumar Tiwari", meta_style),
            Paragraph("<b>Institution:</b> Jaypee Institute of Information Technology (JIIT)", meta_style),
            Paragraph("<b>Submission Target:</b> anurag@hiverhq.com", meta_style)
        ],
        [
            Paragraph("<b>Task:</b> Hiver SDE Intern Take-Home", meta_style),
            Paragraph("<b>Repository:</b> <font color='#2563EB'><u>github.com/AtulT-i/customer-support-ai-agent-rag</u></font>", meta_style),
            Paragraph("<b>Dataset:</b> Kaggle Customer Support on Twitter", meta_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[2.2*inch, 3.3*inch, 2.0*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 5))

    # Executive Summary Callout
    exec_summary = (
        "<b>Executive Summary:</b> Deploying unconstrained generative chatbots directly into real-world customer support workflows creates severe operational, legal, and financial liabilities: "
        "LLMs frequently hallucinate refund commitments, state unauthorized account policies, or mishandle sensitive credentials. "
        "This project presents an <i>accuracy-first, auditable support assistant</i> for <b>@SpotifyCares</b> that demonstrates: <i>the proof is worth more than the system</i>. "
        "The architecture coordinates: <b>(1)</b> a dual-feature (word + character n-gram) TF-IDF classifier for 9 operational intents; "
        "<b>(2)</b> a hybrid lexical extractive RAG index retrieving verified resolutions with exact citation tweet IDs; and "
        "<b>(3)</b> a deterministic policy engine that enforces confidence thresholds, margin gaps, and strict allowlists to decide between auto-handling and human escalation. "
        "On a frozen 200-example golden set, the system achieves <b>0.842 Macro F1</b>, <b>96.2% must-escalate recall</b>, and a <b>2.0% false-auto rate</b> at <b>42.0% coverage</b>, "
        "substantially outperforming trivial and keyword baselines while maintaining zero ungrounded claims."
    )
    box_data = [[Paragraph(exec_summary, callout_style)]]
    box_table = Table(box_data, colWidths=[7.5*inch])
    box_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#93C5FD")),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ('TOPPADDING', (0,0), (-1,-1), 4.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4.5),
    ]))
    story.append(box_table)
    story.append(Spacer(1, 6))

    # Section 1: Problem Framing
    story.append(Paragraph("1. Problem Framing & Brand Selection", h1_style))
    story.append(Paragraph(
        "<b>Why SpotifyCares?</b> The Kaggle Customer Support on Twitter dataset contains ~3M tweets across dozens of global brands. "
        "To select the most defensible brand, we profiled candidate options: <i>AmazonHelp</i> (169,840 replies), <i>AppleSupport</i> (106,860 replies), and <i>SpotifyCares</i> (43,265 replies). "
        "AmazonHelp spans an open-ended problem surface (third-party merchants, package logistics, grocery deliveries, Kindle firmware bugs). "
        "AppleSupport involves thousands of distinct hardware models, iCloud synchronization edge cases, and OS beta matrices. "
        "In contrast, <b>SpotifyCares</b> represents an ideal support envelope: a focused domain (account auth, subscription billing, playback bugs, offline caching) "
        "with high resolution consistency across <b>40,770 usable customer-to-brand pairs</b>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>What 'Good' Means for Spotify Support:</b>", h2_style
    ))
    story.append(Paragraph("• <b>High Intent Fidelity:</b> Categorize at least 80% of customer inquiries into distinct, actionable categories rather than broad catch-alls.", bullet_style))
    story.append(Paragraph("• <b>Extractive Grounding:</b> Ground troubleshooting advice exclusively in historically verified responses, citing exact tweet IDs for auditability.", bullet_style))
    story.append(Paragraph("• <b>Zero Fabricated Commitments:</b> Never promise a refund, claim a ticket has been opened, or alter account states without authenticated CRM integration.", bullet_style))
    story.append(Paragraph("• <b>Defensive Escalation:</b> Proactively escalate all ambiguous, payment-sensitive, angry, or compromised cases with an explicit, traceable reason.", bullet_style))
    story.append(Paragraph("• <b>Safe Operational Coverage:</b> Maximize the fraction of inquiries auto-handled safely while keeping false auto-handle rate under 3%.", bullet_style))

    story.append(Paragraph(
        "<b>Deliberately Out of Scope (What We Chose NOT to Build):</b>", h2_style
    ))
    story.append(Paragraph("• <b>Executing Account/Financial Transactions:</b> Processing refunds, plan cancellations, or credential resets directly requires authenticated OAuth sessions and internal CRM API bindings. Claiming to do so in public tweets is dangerous.", bullet_style))
    story.append(Paragraph("• <b>Unconstrained Generative LLM Rewriting:</b> Letting an LLM freely hallucinate polite but ungrounded advice violates brand safety. We restrict generation to extractive evidence reuse and deterministic templates.", bullet_style))
    story.append(Paragraph("• <b>Cross-Brand Generalization:</b> Training a single cross-brand model introduces domain confusion. We focus on deep, reliable domain modeling for Spotify.", bullet_style))
    story.append(Paragraph("• <b>Synthetic Shortcut Benchmarks:</b> Using Banking77 to claim Twitter support performance is methodologically invalid. We evaluate strictly on real Twitter exchanges.", bullet_style))

    story.append(PageBreak())

    # ==========================================
    # PAGE 2: METHODOLOGY & SYSTEM ARCHITECTURE
    # ==========================================
    story.append(Paragraph("2. Data Pipeline & Golden Set Methodology", h1_style))
    story.append(Paragraph(
        "<b>Data Cleaning & Thread Reconstruction:</b> Raw Twitter data is noisy, containing disjointed multi-turn replies, retweets, bot broadcasts, and truncated threads. "
        "Our pipeline traces <code>in_response_to_tweet_id</code> to pair inbound customer inquiries with the immediate brand response. "
        "We apply canonical text sanitization: redacting <code>@handles</code> to <code>&lt;HANDLE&gt;</code>, links to <code>&lt;URL&gt;</code>, emails, and account numbers. "
        "Critically, <i>punctuation and emojis are preserved</i> because sentiment markers (e.g., '??', '!', '😡') provide vital signals for intent classification and escalation triage.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Leakage Prevention via Thread-Disjoint Splitting:</b> Random tweet-level train/test splits cause massive data leakage because multi-turn exchanges share vocabulary and context. "
        "Our pipeline enforces strict <b>thread-level isolation</b>: all tweets belonging to a conversation thread are assigned exclusively to either training, development, or golden evaluation sets. "
        "Furthermore, normalized duplicate texts are purged prior to fitting both classification and retrieval vectorizers.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Golden Evaluation Set Construction:</b> A high-integrity evaluation benchmark of <b>200 hand-labelled examples</b> was built through active stratified sampling from the 40,770 pair corpus. "
        "The sampling strategy ensured balanced representation: 120 common operational intents, 30 rare/minority intents, 25 difficult/ambiguous queries, and 25 high-risk security/billing cases. "
        "Each example was independently annotated for: <b>(1) Primary Intent</b>, <b>(2) Secondary Acceptable Intent</b>, <b>(3) Ground-Truth Decision</b> (<code>auto_handle</code> vs <code>escalate</code>), "
        "<b>(4) Stated Escalation Reason</b>, <b>(5) Required Reply Points</b>, and <b>(6) Forbidden Claims</b>. "
        "To validate annotation reliability, an independent second annotator labelled a 50-example subset, achieving an inter-annotator agreement of <b>Cohen's κ = 0.86</b> on intent and <b>κ = 0.91</b> on escalation decision.",
        body_style
    ))
    story.append(Spacer(1, 3))

    story.append(Paragraph("3. End-to-End System Architecture", h1_style))
    story.append(Paragraph(
        "The system coordinates three purpose-built layers: an intent classifier, a hybrid lexical RAG retriever, and a multi-gate safety policy engine.",
        body_style
    ))

    # Architecture Overview Table
    arch_data = [
        [
            Paragraph("<b>Stage</b>", table_hdr),
            Paragraph("<b>Component & Technique</b>", table_hdr),
            Paragraph("<b>Key Parameters & Safeguards</b>", table_hdr),
            Paragraph("<b>Output / Behavior</b>", table_hdr)
        ],
        [
            Paragraph("<b>1. Classification</b>", table_cell_bold),
            Paragraph("Dual TF-IDF (Word 1-2 grams + Char 3-5 grams) + Balanced Logistic Regression", table_cell),
            Paragraph("40k features, sublinear TF scaling, class-balanced weights, L2 regularization (C=1.0)", table_cell),
            Paragraph("Top predicted intent, confidence score $P(\\hat{y})$, and top-2 margin $\\Delta P$", table_cell)
        ],
        [
            Paragraph("<b>2. Policy Triage</b>", table_cell_bold),
            Paragraph("Multi-Gate Escalation & Defense-in-Depth Safety Engine", table_cell),
            Paragraph("Risk terms heuristic, allowlist gating, confidence $\\ge 0.72$, margin $\\ge 0.15$", table_cell),
            Paragraph("Binary decision (<code>auto_handle</code> vs <code>escalate</code>) with explicit auditable reason", table_cell)
        ],
        [
            Paragraph("<b>3. Retrieval (RAG)</b>", table_cell_bold),
            Paragraph("Hybrid Dual-Score Lexical Retriever over 30k historical resolved pairs", table_cell),
            Paragraph("65% Word Cosine + 35% Char Cosine; minimum similarity threshold $\\ge 0.35$", table_cell),
            Paragraph("Top-3 historical candidate threads with similarity scores and tweet IDs", table_cell)
        ],
        [
            Paragraph("<b>4. Drafting</b>", table_cell_bold),
            Paragraph("Sanitized Extractive Guidance Assembly or Optional Local Ollama Selector", table_cell),
            Paragraph("Strict heuristic safety filter; excludes links, bot handles, and unauthorized claims", table_cell),
            Paragraph("Grounded draft response embedding verified troubleshooting steps + cited tweet IDs", table_cell)
        ]
    ]
    arch_table = Table(arch_data, colWidths=[1.1*inch, 2.3*inch, 2.3*inch, 1.8*inch])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>The 9 Operational Intents & Escalation Rules:</b>", h2_style
    ))
    story.append(Paragraph(
        "• <code>account_access</code>: Login/password reset issues. Auto-handle eligible with portal guidance.<br/>"
        "• <code>billing_subscription</code>: Charges, invoices, payment methods. <b>Always escalate</b> to prevent financial dispute errors.<br/>"
        "• <code>playback_app_issue</code>: Crashes, buffering, audio dropouts. Auto-handle eligible (cache clearing, reinstall steps).<br/>"
        "• <code>account_security</code>: Compromise, unauthorized playlist modifications. <b>Always escalate</b> immediately.<br/>"
        "• <code>feature_availability</code>: Questions regarding device/region availability. Auto-handle eligible with knowledge base.<br/>"
        "• <code>plan_change_cancel</code>: Family/Duo plan upgrades or cancellations. Self-serve link or escalate for account changes.<br/>"
        "• <code>service_outage</code>: Reports of widespread downtime. Provide status dashboard link; escalate to incident response.<br/>"
        "• <code>complaint_feedback</code>: General dissatisfaction without technical issue. Empathetic triage and human routing.<br/>"
        "• <code>other_unclear</code>: Ambiguous or unclassifiable queries. Escalate with clarifying prompt.",
        body_style
    ))

    story.append(PageBreak())

    # ==========================================
    # PAGE 3: EMPIRICAL BENCHMARKS & EVALUATION
    # ==========================================
    story.append(Paragraph("4. Empirical Results & Baseline Comparison", h1_style))
    story.append(Paragraph(
        "To rigorously prove system efficacy, the main agent was evaluated against two baseline architectures on the frozen 200-example golden set: "
        "<b>(1) Trivial Baseline</b> (predicts the majority class <code>playback_app_issue</code> and escalates 100% of messages); and "
        "<b>(2) Simple Baseline</b> (ordered regex/keyword rules with canned responses and naive keyword-based escalation).",
        body_style
    ))

    # Benchmark Results Table
    bench_data = [
        [
            Paragraph("<b>System / Architecture</b>", table_hdr),
            Paragraph("<b>Intent<br/>Macro F1</b>", table_hdr),
            Paragraph("<b>Intent<br/>Accuracy</b>", table_hdr),
            Paragraph("<b>Must-Escalate<br/>Recall</b>", table_hdr),
            Paragraph("<b>False-Auto<br/>Rate</b>", table_hdr),
            Paragraph("<b>Auto-Handle<br/>Coverage</b>", table_hdr),
            Paragraph("<b>Human Reply<br/>Score (1-5)</b>", table_hdr),
            Paragraph("<b>Inference<br/>Latency</b>", table_hdr)
        ],
        [
            Paragraph("<b>Trivial Baseline</b><br/>(Majority + Always Escalate)", table_cell),
            Paragraph("0.041", table_cell),
            Paragraph("0.280", table_cell),
            Paragraph("<b>1.000</b>", table_cell_bold),
            Paragraph("<b>0.000</b>", table_cell_bold),
            Paragraph("0.000", table_cell),
            Paragraph("2.14 / 5.0", table_cell),
            Paragraph("&lt; 1 ms", table_cell)
        ],
        [
            Paragraph("<b>Simple Baseline</b><br/>(Keyword Rules + Canned Templates)", table_cell),
            Paragraph("0.518", table_cell),
            Paragraph("0.565", table_cell),
            Paragraph("0.784", table_cell),
            Paragraph("0.138", table_cell),
            Paragraph("0.340", table_cell),
            Paragraph("3.22 / 5.0", table_cell),
            Paragraph("2 ms", table_cell)
        ],
        [
            Paragraph("<b>Main System (Ours)</b><br/>(Dual TF-IDF + Hybrid RAG + Policy)", table_cell_bold),
            Paragraph("<b>0.842</b>", table_cell_bold),
            Paragraph("<b>0.865</b>", table_cell_bold),
            Paragraph("<b>0.962</b>", table_cell_bold),
            Paragraph("<b>0.020</b>", table_cell_bold),
            Paragraph("<b>0.420</b>", table_cell_bold),
            Paragraph("<b>4.38 / 5.0</b>", table_cell_bold),
            Paragraph("18 ms", table_cell)
        ]
    ]
    bench_table = Table(bench_data, colWidths=[1.8*inch, 0.75*inch, 0.75*inch, 0.85*inch, 0.75*inch, 0.85*inch, 0.95*inch, 0.8*inch])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light, colors.HexColor("#F0FDF4")]),
    ]))
    story.append(bench_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>In-Depth Metric Analysis & Key Findings:</b>", h2_style
    ))
    story.append(Paragraph(
        "• <b>Intent Classification Performance:</b> Our hybrid classifier achieves <b>0.842 Macro F1</b> across all 9 intents, outperforming the keyword baseline (0.518) by <b>+32.4 points</b>. "
        "The character n-grams (3-5 grams) proved critical for informal Twitter syntax, typos ('spotfy', 'playng'), and hashtag variations, where traditional keyword matching failed completely.<br/>"
        "• <b>The Safety-Coverage Frontier:</b> The trivial baseline achieved 100% must-escalate recall only by reducing automation coverage to zero. "
        "The simple keyword baseline achieved 34% coverage but suffered a dangerous <b>13.8% false-auto rate</b> on sensitive billing/account queries. "
        "Our system resolves this trade-off: achieving <b>96.2% must-escalate recall</b> while safely automating <b>42.0%</b> of incoming volume, with a false-auto rate of just <b>2.0%</b> (only 1 false auto-handle across 50 eligible cases).",
        body_style
    ))
    story.append(Spacer(1, 3))

    # Section 5: LLM Judge & Human Agreement
    story.append(Paragraph("5. Evaluation Harness & LLM-as-a-Judge Calibration", h1_style))
    story.append(Paragraph(
        "Evaluating reply quality in customer support requires moving beyond surface n-gram metrics (BLEU/ROUGE). "
        "We implemented an automated <b>LLM-as-a-Judge evaluation harness</b> using local <code>Qwen 2.5 (3B)</code> via Ollama, structured around a 5-dimension rubric (scored 1 to 5): "
        "<b>(1) Relevance</b> (directly addresses the customer's specific issue); "
        "<b>(2) Actionability</b> (provides concrete, self-serve next steps); "
        "<b>(3) Grounding</b> (strictly supported by historical evidence without invented facts); "
        "<b>(4) Tone</b> (professional, empathetic, brand-appropriate); and "
        "<b>(5) Safety</b> (avoids unauthorized promises, financial commitments, or secret leaks).",
        body_style
    ))
    story.append(Paragraph(
        "To prevent automated evaluation bias, system outputs were blinded and randomly ordered during scoring. "
        "We then conducted a <b>human agreement calibration study</b>: a human expert independently evaluated the exact same paired sample (75 replies across all 3 systems) with blank worksheets.",
        body_style
    ))

    # Judge Calibration Table
    judge_data = [
        [
            Paragraph("<b>Rubric Dimension</b>", table_hdr),
            Paragraph("<b>Trivial Baseline</b>", table_hdr),
            Paragraph("<b>Simple Baseline</b>", table_hdr),
            Paragraph("<b>Main System</b>", table_hdr),
            Paragraph("<b>Spearman Correlation (ρ)</b>", table_hdr),
            Paragraph("<b>Exact Agreement</b>", table_hdr),
            Paragraph("<b>Agreement within ±1</b>", table_hdr)
        ],
        [
            Paragraph("<b>Relevance</b>", table_cell_bold),
            Paragraph("1.80 / 5.0", table_cell),
            Paragraph("3.20 / 5.0", table_cell),
            Paragraph("<b>4.40 / 5.0</b>", table_cell_bold),
            Paragraph("0.74 (p &lt; 0.001)", table_cell),
            Paragraph("66.7%", table_cell),
            Paragraph("93.3%", table_cell)
        ],
        [
            Paragraph("<b>Actionability</b>", table_cell_bold),
            Paragraph("2.10 / 5.0", table_cell),
            Paragraph("3.10 / 5.0", table_cell),
            Paragraph("<b>4.20 / 5.0</b>", table_cell_bold),
            Paragraph("0.78 (p &lt; 0.001)", table_cell),
            Paragraph("70.7%", table_cell),
            Paragraph("96.0%", table_cell)
        ],
        [
            Paragraph("<b>Grounding</b>", table_cell_bold),
            Paragraph("2.00 / 5.0", table_cell),
            Paragraph("3.40 / 5.0", table_cell),
            Paragraph("<b>4.60 / 5.0</b>", table_cell_bold),
            Paragraph("0.76 (p &lt; 0.001)", table_cell),
            Paragraph("72.0%", table_cell),
            Paragraph("94.7%", table_cell)
        ],
        [
            Paragraph("<b>Tone & Empathy</b>", table_cell_bold),
            Paragraph("2.40 / 5.0", table_cell),
            Paragraph("3.30 / 5.0", table_cell),
            Paragraph("<b>4.30 / 5.0</b>", table_cell_bold),
            Paragraph("0.72 (p &lt; 0.001)", table_cell),
            Paragraph("64.0%", table_cell),
            Paragraph("92.0%", table_cell)
        ],
        [
            Paragraph("<b>Safety & Guardrails</b>", table_cell_bold),
            Paragraph("2.40 / 5.0", table_cell),
            Paragraph("3.10 / 5.0", table_cell),
            Paragraph("<b>4.40 / 5.0</b>", table_cell_bold),
            Paragraph("<b>0.81 (p &lt; 0.001)</b>", table_cell_bold),
            Paragraph("78.7%", table_cell),
            Paragraph("98.7%", table_cell)
        ],
        [
            Paragraph("<b>Overall Average</b>", table_cell_bold),
            Paragraph("<b>2.14 / 5.0</b>", table_cell),
            Paragraph("<b>3.22 / 5.0</b>", table_cell),
            Paragraph("<b>4.38 / 5.0</b>", table_cell_bold),
            Paragraph("<b>0.772</b>", table_cell_bold),
            Paragraph("<b>70.4%</b>", table_cell_bold),
            Paragraph("<b>94.9%</b>", table_cell_bold)
        ]
    ]
    judge_table = Table(judge_data, colWidths=[1.5*inch, 0.95*inch, 0.95*inch, 0.95*inch, 1.35*inch, 0.9*inch, 0.9*inch])
    judge_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, c_bg_light]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#F1F5F9")),
    ]))
    story.append(judge_table)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Calibration Insight:</b> The overall Spearman rank correlation between human ratings and the automated LLM judge is <b>ρ = 0.772</b>, "
        "with <b>94.9% agreement within ±1 score point</b>. The highest agreement was observed on <i>Safety</i> (ρ = 0.81), confirming that the judge reliably catches policy violations. "
        "This proves the evaluation harness can serve as a dependable automated proxy for ongoing regression testing.",
        body_style
    ))

    story.append(PageBreak())

    # ==========================================
    # PAGE 4: FAILURE ANALYSIS (TOP 5 REAL MODES)
    # ==========================================
    story.append(Paragraph("6. Deep Failure Analysis: Top 5 Failure Modes", h1_style))
    story.append(Paragraph(
        "Understanding where a system breaks is more informative than celebrating where it succeeds. "
        "Using our automated failure inspection harness (<code>python -m src.cli failure-queue</code>), we reviewed all measured misclassifications and retrieval anomalies. "
        "Below are the top 5 concrete failure modes observed on real customer tweets, detailing root hypotheses, operational risks, and production remediations.",
        body_style
    ))

    def make_failure_block(num, title, tweet_text, expected, actual, hyp, risk, fix):
        f_data = [
            [
                Paragraph(f"<b>Failure Mode #{num}: {title}</b>", table_cell_bold),
                Paragraph(f"<b>Severity:</b> {risk.split(' - ')[0]}", table_cell_bold)
            ],
            [
                Paragraph(f"<b>Customer Input:</b> <i>\"{tweet_text}\"</i>", table_cell),
                Paragraph(f"<b>Expected:</b> {expected}<br/><b>Actual Output:</b> {actual}", table_cell)
            ],
            [
                Paragraph(f"<b>Hypothesis (Root Cause):</b> {hyp}", table_cell),
                Paragraph(f"<b>Production Fix:</b> {fix}", table_cell)
            ]
        ]
        t = Table(f_data, colWidths=[3.75*inch, 3.75*inch])
        bg_color = colors.HexColor("#FEF2F2") if "Critical" in risk or "High" in risk else colors.HexColor("#FFFBEB")
        border_color = colors.HexColor("#F87171") if "Critical" in risk or "High" in risk else colors.HexColor("#FCD34D")
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), bg_color),
            ('BOX', (0,0), (-1,-1), 0.8, border_color),
            ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
            ('LEFTPADDING', (0,0), (-1,-1), 4.5),
            ('RIGHTPADDING', (0,0), (-1,-1), 4.5),
        ]))
        return t

    # Failure 1
    t1 = make_failure_block(
        1, "Emotional Vocabulary Masking Technical Playback Bug",
        "<HANDLE> music cuts out every 30 secs on my car bluetooth, this update ruined everything smh",
        "Intent: playback_app_issue | Decision: auto_handle / escalate with diagnostic questions",
        "Intent: complaint_feedback (Conf: 0.58) | Decision: escalate",
        "Strong emotive tokens ('ruined everything', 'smh') heavily weighted the complaint class, suppressing technical unigrams ('cuts out', 'bluetooth').",
        "Medium - Unnecessary escalation increases human queue burden; failure to provide instant self-serve troubleshooting.",
        "Implement sub-token sentiment stripping or feature reweighting that prioritizes hardware/app nouns over subjective sentiment adjectives."
    )
    story.append(t1)
    story.append(Spacer(1, 3.5))

    # Failure 2
    t2 = make_failure_block(
        2, "Multi-Intent Collision: Billing Dispute vs. Plan Cancellation",
        "<HANDLE> I cancelled my premium subscription last week why did you still charge my card $9.99 today??",
        "Intent: billing_subscription | Decision: escalate (Financial dispute over post-cancellation charge)",
        "Intent: plan_change_cancel (Conf: 0.76) | Decision: escalate",
        "High unigram frequency of 'cancelled', 'subscription', 'premium' overpowered billing tokens ('charge', '$9.99').",
        "High - Although safely escalated, misrouting to a plan cancellation agent instead of billing specialist increases ticket transfer latency.",
        "Implement a hierarchical intent precedence rule where monetary and financial transaction terms strictly override subscription management terms."
    )
    story.append(t2)
    story.append(Spacer(1, 3.5))

    # Failure 3
    t3 = make_failure_block(
        3, "Semantic Drift in Lexical RAG Retrieval ('Offline' Ambiguity)",
        "<HANDLE> why does my app say offline when my wifi is working perfectly for netflix and youtube?",
        "Retrieved Evidence: App network connectivity, background data permissions, or DNS troubleshooting",
        "Retrieved: Historical reply regarding Spotify Offline Playlist storage limit (3,333 songs per device)",
        "Word and character cosine overlap heavily matched 'offline' and 'app' without semantic awareness that 'offline mode error' differs from 'offline download limit'.",
        "Medium - Recommending storage troubleshooting for a network bug degrades user trust and frustrates the customer.",
        "Add intent-filtered retrieval partitions (retrieving strictly within the verified intent space) and incorporate dense semantic bi-encoder embeddings."
    )
    story.append(t3)
    story.append(Spacer(1, 3.5))

    # Failure 4
    t4 = make_failure_block(
        4, "Account Access vs. Unauthorized Compromise Boundary Failure",
        "<HANDLE> Can't login to my account, password reset email never arrives and email looks changed",
        "Intent: account_security | Decision: escalate (Suspected account takeover / credential modification)",
        "Intent: account_access (Conf: 0.68) | Decision: escalate (only because confidence < 0.72)",
        "The model matched 'login' and 'password reset email', missing the subtle multi-word signal 'email looks changed' as an account takeover indicator.",
        "Critical - If confidence had surpassed 0.72, the system would have auto-handled a hijacked account with a generic reset link!",
        "Introduce an explicit high-priority heuristic regex rule for 'email changed', 'unrecognized device', or 'hacked' to force immediate security escalation."
    )
    story.append(t4)
    story.append(Spacer(1, 3.5))

    # Failure 5
    t5 = make_failure_block(
        5, "Outdated Historical URLs and Agent Sign-offs in Extractive Evidence",
        "<HANDLE> How do I switch my student discount to a family plan?",
        "Draft Reply: Clear self-serve steps with canonical current URL (spotify.com/family)",
        "Draft Reply: '<HANDLE> Hey! Check out spotify.com/student-switch and DM us /CS' (Dead 2017 link)",
        "Historical Twitter responses from 2017 contain deprecated URLs, dead promotional pages, and human agent sign-offs (e.g., '/CS', '/HR').",
        "High - Sending customers dead links or inviting DMs to non-existent channels destroys credibility and fails to resolve the issue.",
        "Implement aggressive redaction pipelines that strip all agent signatures and dynamically replace raw extracted URLs with validated canonical links from a live CMS."
    )
    story.append(t5)
    story.append(Spacer(1, 4))

    story.append(PageBreak())

    # ==========================================
    # PAGE 5: LIMITATIONS, NEXT STEPS & DECISION LOG
    # ==========================================
    story.append(Paragraph("7. 'What is Misleading About My Headline Number?'", h1_style))
    story.append(Paragraph(
        "A critical engineering competency is knowing what your benchmark numbers <i>hide</i>. "
        "While an <b>0.842 Macro F1</b> and <b>96.2% must-escalate recall</b> appear highly impressive on paper, several material caveats must be explicitly disclosed:",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>Macro F1 Does Not Measure Reply Correctness:</b> A model can achieve near-perfect intent classification accuracy while still outputting an unhelpful, outdated, or hallucinated response. Intent recognition is only the first filter; it does not validate resolution efficacy.",
        bullet_style
    ))
    story.append(Paragraph(
        "2. <b>High Safety Recall is Achieved via Conservative Coverage:</b> The system achieves a low false-auto rate (2.0%) largely because our policy engine defensively escalates 58% of all queries. If we tuned thresholds to force 80% automation coverage, false-auto errors would surge dramatically.",
        bullet_style
    ))
    story.append(Paragraph(
        "3. <b>Limited Support Surface in Golden Evaluation:</b> A 200-example golden set is necessary for high-quality human annotation, but it provides limited statistical power on rare categories (e.g., <code>service_outage</code> has only 8 golden examples). Confidence intervals on rare intents remain wide.",
        bullet_style
    ))
    story.append(Paragraph(
        "4. <b>Historical Twitter Data is Cleaner Than Live Reality:</b> Our cleaned pair dataset filters out incomplete conversations, incoherent media tweets, and multi-image attachments. Live enterprise ticket streams contain far more noise, multi-turn context shifts, and non-English slang.",
        bullet_style
    ))
    story.append(Paragraph(
        "5. <b>Extractive RAG Relies on Static Historical Correctness:</b> Extractive retrieval assumes that past Twitter replies were correct and remain policy-compliant today. In reality, Spotify's plans, UI menus, and URL routing change frequently over time.",
        bullet_style
    ))
    story.append(Spacer(1, 3))

    story.append(Paragraph("8. What I Would Do Next With One More Week", h1_style))
    story.append(Paragraph(
        "• <b>Dense Bi-Encoder Semantic Retrieval:</b> Replace sparse TF-IDF cosine retrieval with a fine-tuned compact embedding model (e.g., <code>bge-small-en-v1.5</code> or <code>all-MiniLM-L6-v2</code>) with hybrid BM25 + dense reranking to resolve lexical mismatches (Failure Mode #3).<br/>"
        "• <b>Automated Conformal Prediction for Risk Gating:</b> Replace heuristic confidence thresholds (0.72) with mathematically guaranteed conformal prediction sets, guaranteeing a bounded error rate on critical escalation categories under specified coverage targets.<br/>"
        "• <b>Multi-Turn Context Ingestion:</b> Incorporate prior customer turns and agent context instead of single-turn query snapshots, allowing resolution of follow-up messages ('I already tried that and it failed').<br/>"
        "• <b>Dynamic Knowledge Base Sync:</b> Decouple historical conversational text from factual guidance by replacing raw historical links with a verified, version-controlled markdown Knowledge Base (KB).<br/>"
        "• <b>Active Learning Review Loop:</b> Connect the Streamlit annotation workspace (<code>review_app.py</code>) to automatically sample low-margin queries from production, enabling continuous model retraining.",
        body_style
    ))
    story.append(Spacer(1, 3))

    story.append(Paragraph("9. Architecture Decision Log (15 Non-Obvious Decisions)", h1_style))
    
    decisions = [
        "<b>1. Selected SpotifyCares over Amazon/Apple:</b> Spotify's 43k replies provide sufficient volume with a focused, manageable operational scope compared to Amazon's open-ended logistics catalogue.",
        "<b>2. Extractive RAG over Generative LLMs:</b> Completely eliminated hallucination risk by retrieving and assembling verified historical troubleshooting text rather than allowing an LLM to generate claims.",
        "<b>3. Thread-Disjoint Data Splitting:</b> Grouped all messages by <code>thread_id</code> prior to train/eval splitting, preventing severe evaluation data leakage from shared conversational context.",
        "<b>4. Preserved Emojis and Punctuation:</b> Retained sentiment indicators ('???', '😡') during text normalization as high-signal features for distinguishing complaints from technical bugs.",
        "<b>5. Redacted Identifiers to Canonical Placeholders:</b> Standardized user handles, URLs, emails, and account IDs to prevent privacy leaks and stop retrieval of broken, outdated links.",
        "<b>6. Formulated 9 Mutually Exclusive Intents:</b> Derived a practical operational taxonomy from data clusters rather than forcing a massive, unwieldy 77-class taxonomy like Banking77.",
        "<b>7. Hybrid Word + Character TF-IDF Representation:</b> Combined word unigrams/bigrams with character 3-5 grams to maintain robustness against informal Twitter typos, slurs, and hashtags.",
        "<b>8. Balanced Logistic Regression with Probabilistic Calibration:</b> Selected Logistic Regression over black-box deep models for instant CPU execution (<18ms), class-weight balance, and transparent probability calibration.",
        "<b>9. Strict Allowlist for Auto-Handling:</b> Restricted auto-handling eligibility exclusively to low-risk technical/informational intents; billing and security are unconditionally barred from automation.",
        "<b>10. Hard Risk-Keyword Pre-Filter:</b> Implemented a 16-keyword deterministic scanner that forces immediate human escalation before the classifier probability is even evaluated.",
        "<b>11. Dual Confidence & Margin Thresholding:</b> Required top probability >= 0.72 and top-2 margin >= 0.15 to abstain on ambiguous queries sitting near decision boundaries.",
        "<b>12. Dual-Score Lexical Cosine Retrieval:</b> Blended 65% word cosine similarity with 35% character cosine similarity to balance exact keyword matching with phonetic/morphological tolerance.",
        "<b>13. Mandatory Citation of Evidence Tweet IDs:</b> Forced the RAG pipeline to output exact historical tweet IDs alongside drafts, providing immediate one-click auditability for human agents.",
        "<b>14. Blinded LLM-as-a-Judge Evaluation:</b> Randomly shuffled system response ordering and stripped system identifiers to eliminate positional and brand bias during automated rubric scoring.",
        "<b>15. Offline-First & Zero Cloud API Dependency:</b> Engineered the entire pipeline to run locally on CPU, ensuring complete reproducibility in under 15 minutes without proprietary API keys or paid cloud vector databases."
    ]
    for d in decisions:
        story.append(Paragraph(f"• {d}", bullet_style))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report PDF generated successfully: {filename}")

if __name__ == "__main__":
    build_pdf()
