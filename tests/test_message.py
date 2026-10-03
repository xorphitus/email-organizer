from email_organizer.message import clean_body, extract_text, html_to_text, message_key, parse_rfc822


def test_html_to_text_strips_tags_and_scripts():
    html = "<html><head><style>p{}</style></head><body><p>Hello&nbsp;<b>world</b></p><script>x()</script><div>Bye</div></body></html>"
    assert html_to_text(html) == "Hello world\n\nBye"


def test_clean_body_trims_quotes_and_signature():
    text = "Thanks!\n\nSee you.\n-- \nAlice\nCEO\n"
    assert clean_body(text) == "Thanks!\n\nSee you."
    reply = "Sounds good.\n\nOn Mon, 1 Oct 2026, Bob <b@x> wrote:\n> earlier\n> text"
    assert clean_body(reply) == "Sounds good."
    inline = "Yes\n> quoted\nNo"
    assert clean_body(inline) == "Yes\nNo"


def test_clean_body_keeps_leading_from_line():
    fwd = "From: shop@example.com\nYour order shipped."
    assert "Your order shipped." in clean_body(fwd)


def test_clean_body_truncates():
    assert clean_body("x" * 100, max_chars=10) == "x" * 10 + " …"


RAW_ALT = b"""From: Shop <shop@example.com>
To: me@example.com
Subject: Your receipt
Message-ID: <abc@example.com>
List-Id: <shop.example.com>
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="b"

--b
Content-Type: text/plain; charset=utf-8

Plain body
--b
Content-Type: text/html; charset=utf-8

<p>HTML body</p>
--b--
"""

RAW_HTML = b"""From: a@example.com
Subject: hi
Content-Type: text/html; charset=utf-8

<p>Only <i>html</i></p>
"""


def test_extract_text_prefers_plain():
    msg = parse_rfc822(RAW_ALT)
    assert extract_text(msg).strip() == "Plain body"
    assert message_key(msg) == "<abc@example.com>"


def test_extract_text_html_fallback_and_hash_key():
    msg = parse_rfc822(RAW_HTML)
    assert extract_text(msg) == "Only html"
    assert message_key(msg).startswith("sha256:")
