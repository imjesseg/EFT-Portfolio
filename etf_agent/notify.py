"""Email a Markdown report through any SMTP server, configured with environment variables."""

from __future__ import annotations

import html
import os
import re
import smtplib
import ssl
from email.message import EmailMessage

CELL = "border:1px solid #ccc;padding:4px 8px;"


def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"^_(.+)_$", r"<em>\1</em>", text)


def markdown_to_html(markdown: str) -> str:
    """Just enough Markdown for this tool's reports: headings, paragraphs, bullets, quotes, tables, links, bold."""
    out: list[str] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            header, separator, body = rows[0], rows[1], rows[2:]
            align = ["right" if s.endswith(":") else "left" for s in separator]
            out.append('<table style="border-collapse:collapse;font-size:14px;margin:8px 0;">')
            out.append("<tr>" + "".join(f'<th style="{CELL}text-align:{a};background:#f3f3f3;">{_inline(c)}</th>' for c, a in zip(header, align)) + "</tr>")
            for row in body:
                out.append("<tr>" + "".join(f'<td style="{CELL}text-align:{a};">{_inline(c)}</td>' for c, a in zip(row, align)) + "</tr>")
            out.append("</table>")
            continue
        if line.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("</ul>")
            continue
        heading = re.match(r"(#{1,3}) (.*)", line)
        if heading:
            out.append(f"<h{len(heading.group(1))}>{_inline(heading.group(2))}</h{len(heading.group(1))}>")
        elif line.startswith("> "):
            out.append(f'<blockquote style="border-left:3px solid #ccc;margin:8px 0;padding-left:8px;">{_inline(line[2:])}</blockquote>')
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    return '<html><body style="font-family:Arial,sans-serif;">' + "\n".join(out) + "</body></html>"


def send_email(subject: str, markdown: str) -> None:
    """Send ``markdown`` as plain text plus HTML.

    Configure with SMTP_HOST, SMTP_PORT (default 587; 465 uses SSL), SMTP_USER, SMTP_PASSWORD,
    EMAIL_TO (comma-separated), and optionally EMAIL_FROM (defaults to SMTP_USER).
    """
    env = os.environ
    host, to = env.get("SMTP_HOST"), env.get("EMAIL_TO")
    sender = env.get("EMAIL_FROM") or env.get("SMTP_USER")
    if not (host and to and sender):
        raise RuntimeError("Set SMTP_HOST, EMAIL_TO, and SMTP_USER or EMAIL_FROM to send email")
    message = EmailMessage()
    message["Subject"], message["From"], message["To"] = subject, sender, to
    message.set_content(markdown)
    message.add_alternative(markdown_to_html(markdown), subtype="html")

    port = int(env.get("SMTP_PORT", "587"))
    context = ssl.create_default_context()
    smtp = smtplib.SMTP_SSL(host, port, context=context, timeout=30) if port == 465 else smtplib.SMTP(host, port, timeout=30)
    with smtp:
        smtp.ehlo()
        if port != 465 and smtp.has_extn("starttls"):
            smtp.starttls(context=context)
        if env.get("SMTP_USER") and env.get("SMTP_PASSWORD"):
            smtp.login(env["SMTP_USER"], env["SMTP_PASSWORD"])
        smtp.send_message(message)
