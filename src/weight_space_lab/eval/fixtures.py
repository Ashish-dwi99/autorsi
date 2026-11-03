"""Generated local-only tasks with fixture-owned success conditions.

Development and acceptance share task families but use different fixed seeds.
They test elementary interactions on synthetic pages, not unfamiliar real sites,
desktop control, security robustness, or broad computer-use capability.  Reserve
acceptance policy executions until a candidate has been selected on development.
"""

from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
import random


@dataclass(frozen=True)
class BrowserTask:
    task_id: str
    split: str
    family: str
    goal: str
    html: str
    allowed_ids: tuple[str, ...]
    reference_actions: tuple[dict, ...]

    @property
    def fixture_sha256(self) -> str:
        return sha256(self.html.encode("utf-8")).hexdigest()


def _document(title: str, body: str, script: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'">
<title>{escape(title)}</title>
<style>
*{{box-sizing:border-box}}body{{font:18px system-ui;margin:30px;color:#15243a;background:#f5f7fb}}
main{{max-width:900px;background:white;padding:28px;border:1px solid #ccd5df;border-radius:12px}}
h1{{margin-top:0;font-size:27px}}label{{display:block;margin:15px 0}}input,select,button{{font:inherit;padding:9px;margin:5px}}
input[type=checkbox],input[type=radio]{{width:20px;height:20px}}button{{background:#e3edfa;border:1px solid #8ba6c9;border-radius:5px;cursor:pointer}}
.row{{padding:12px;border-bottom:1px solid #dae0e8}}.muted{{color:#526176}}[hidden]{{display:none!important}}
#result{{min-height:30px;padding-top:15px;font-weight:bold}}.dialog{{padding:20px;border:2px solid #496f9c;background:#f5faff}}
</style></head><body><main><h1>{escape(title)}</h1>{body}
<p id="result" data-success="false" role="status"></p></main>
<script>
const byId=(id)=>document.querySelector('[data-id="'+id+'"]');
const result=document.querySelector('#result');
const finish=(ok)=>{{result.dataset.success=String(Boolean(ok));result.textContent=ok?'Saved successfully.':'The requested settings are not correct yet.';}};
{script}
</script></body></html>"""


def _button(i: str, label: str) -> str:
    return f'<button type="button" data-id="{i}">{escape(label)}</button>'


def _action(kind: str, element_id: str | None = None, **kw) -> dict:
    return {"action": kind, **({"id": element_id} if element_id else {}), **kw}


def _task(split: str, family: str, goal: str, title: str, body: str,
          script: str, ids: list[str], actions: list[dict]) -> BrowserTask:
    return BrowserTask(f"{split}-{family}", split, family, goal,
                       _document(title, body, script), tuple(ids), tuple(actions))


def build_suite(split: str) -> tuple[BrowserTask, ...]:
    """Return exactly eight deterministic fixtures without browser execution."""
    if split not in {"development", "acceptance"}:
        raise ValueError("split must be development or acceptance")
    rng = random.Random(2026110301 if split == "development" else 2026110302)
    suffix = rng.randint(100, 999)
    name = rng.choice(["Mira", "Jonah", "Avery", "Tessa"]) + f" {suffix}"
    cities = ["Bristol", "Dover", "Leeds", "York"]
    rng.shuffle(cities)
    city = cities[1]
    tasks = []

    # Four interactions plus submit, with a genuine checkbox and select.
    body = ('<p>Create the visitor record. All fields must match the request.</p>'
            '<label>Visitor name <input data-id="e1" aria-label="Visitor name"></label>'
            '<label>City <select data-id="e2" aria-label="City"><option value="">Choose city</option>'
            + ''.join(f'<option value="{x}">{x}</option>' for x in cities)
            + '</select></label><label><input type="checkbox" data-id="e3"> Email receipt</label>'
            + _button("e4", "Save visitor"))
    script = ("byId('e4').onclick=()=>finish(byId('e1').value===" + json.dumps(name)
              + " && byId('e2').value===" + json.dumps(city) + " && byId('e3').checked);")
    tasks.append(_task(split, "visitor", f"Save visitor {name} in {city}, with Email receipt enabled.",
                       "Visitor registration", body, script, ["e1", "e2", "e3", "e4"],
                       [_action("type", "e1", text=name), _action("select", "e2", value=city),
                        _action("click", "e3"), _action("click", "e4")]))

    # Numeric values are on the page; the goal contains no answer.
    a, b = rng.randint(14, 49), rng.randint(11, 39)
    body = (f'<p>Morning deliveries: <strong>{a}</strong></p><p>Afternoon deliveries: <strong>{b}</strong></p>'
            '<label>Total deliveries <input data-id="e1" inputmode="numeric" aria-label="Total deliveries"></label>'
            + _button("e2", "Submit total"))
    script = f"byId('e2').onclick=()=>finish(byId('e1').value.trim()==={json.dumps(str(a+b))});"
    tasks.append(_task(split, "addition", "Enter the combined morning and afternoon delivery count and submit it.",
                       "Delivery totals", body, script, ["e1", "e2"],
                       [_action("type", "e1", text=str(a+b)), _action("click", "e2")]))

    # Click the requested row then choose an action in its revealed panel.
    products = (["Cedar lamp", "Birch lamp", "Maple lamp", "Oak lamp"] if split == "development"
                else ["Amber clock", "Ivory clock", "Slate clock", "Copper clock"])
    rng.shuffle(products)
    desired_index = rng.randrange(len(products))
    desired = products[desired_index]
    body = '<p>Open a product, then reserve it. Reserve only the requested product.</p>'
    script = "let selected=null;const reserved=[];"
    for n, product in enumerate(products, 1):
        body += f'<div class="row">{escape(product)} {_button(f"e{n}", "View " + product)}</div>'
        script += f"byId('e{n}').onclick=()=>{{selected={json.dumps(product)};document.querySelector('#detail').hidden=false;document.querySelector('#selected').textContent=selected;}};"
    body += ('<section id="detail" hidden><h2 id="selected"></h2>' + _button("e5", "Reserve this product") + '</section>')
    script += "byId('e5').onclick=()=>{reserved.push(selected);finish(reserved.length===1&&reserved[0]===" + json.dumps(desired) + ");};"
    tasks.append(_task(split, "product", f"Reserve the {desired}.", "Product catalog", body, script,
                       ["e1", "e2", "e3", "e4", "e5"], [_action("click", f"e{desired_index+1}"), _action("click", "e5")]))

    # Deactivate an existing choice, enable another, and save.
    labels = (["Daily summary", "Weekly digest", "Promotional offers"] if split == "development"
              else ["Delivery updates", "Monthly report", "Product surveys"])
    rng.shuffle(labels)
    on_index, off_index, unchanged_index = rng.sample(range(3), 3)
    turn_on, turn_off = labels[on_index], labels[off_index]
    body = '<p>Change the notification settings and save.</p>'
    for n, label in enumerate(labels, 1):
        checked = ' checked' if n == off_index + 1 else ''
        body += f'<label><input type="checkbox" data-id="e{n}"{checked}> {label}</label>'
    body += _button("e4", "Save preferences")
    script = f"byId('e4').onclick=()=>finish(byId('e{on_index+1}').checked&&!byId('e{off_index+1}').checked&&!byId('e{unchanged_index+1}').checked);"
    tasks.append(_task(split, "preferences", f"Enable {turn_on}, disable {turn_off}, leave {labels[unchanged_index]} disabled, and save.",
                       "Notification preferences", body, script, ["e1", "e2", "e3", "e4"],
                       [_action("click", f"e{on_index+1}"), _action("click", f"e{off_index+1}"), _action("click", "e4")]))

    # Search and choose among similar labels; matching requires the exact row.
    prefix = "R" if split == "development" else "T"
    code = prefix + str(rng.randint(200, 899))
    codes = [code[:-1] + str((int(code[-1])+1)%10), code, prefix + str(rng.randint(900, 999))]
    rng.shuffle(codes)
    body = '<label>Search records <input data-id="e1" aria-label="Search records"></label>'
    body += _button("e2", "Search") + '<div id="records">'
    for n, record in enumerate(codes, 3):
        body += f'<div class="row" data-record="{record}">Record {record} {_button(f"e{n}", "Open " + record)}</div>'
    body += '</div><section id="record-detail" hidden><p id="record-name"></p>' + _button("e6", "Mark reviewed") + '</section>'
    script = "let selected=null;const reviewed=[];byId('e2').onclick=()=>{const q=byId('e1').value;document.querySelectorAll('[data-record]').forEach(e=>e.hidden=!e.dataset.record.includes(q));};"
    for n, record in enumerate(codes, 3):
        script += f"byId('e{n}').onclick=()=>{{selected={json.dumps(record)};document.querySelector('#record-name').textContent=selected;document.querySelector('#record-detail').hidden=false;}};"
    script += "byId('e6').onclick=()=>{reviewed.push(selected);finish(reviewed.length===1&&reviewed[0]===" + json.dumps(code) + ");};"
    tasks.append(_task(split, "record", f"Find record {code} and mark it reviewed.", "Record archive", body, script,
                       [f"e{x}" for x in range(1, 7)],
                       [_action("type", "e1", text=code), _action("click", "e2"), _action("click", f"e{codes.index(code)+3}"), _action("click", "e6")]))

    # An initially hidden panel enforces an observation/action transition.
    pin = str(rng.randint(1000, 4999) if split == "development" else rng.randint(6000, 9999))
    body = (f'<p>Your pickup code is <strong>{pin}</strong>.</p>' + _button("e1", "Open pickup form")
            + '<section class="dialog" id="pickup" hidden><h2>Pickup confirmation</h2>'
              '<label>Pickup code <input data-id="e2" aria-label="Pickup code"></label>'
            + _button("e3", "Confirm pickup") + _button("e4", "Cancel") + '</section>')
    script = ("byId('e1').onclick=()=>document.querySelector('#pickup').hidden=false;"
              "byId('e4').onclick=()=>document.querySelector('#pickup').hidden=true;"
              "byId('e3').onclick=()=>finish(byId('e2').value===" + json.dumps(pin) + ");")
    tasks.append(_task(split, "pickup", "Open the pickup form and confirm pickup using the displayed code.",
                       "Parcel pickup", body, script, ["e1", "e2", "e3", "e4"],
                       [_action("click", "e1"), _action("type", "e2", text=pin), _action("click", "e3")]))

    # Two-stage choice with distractor tab and select options.
    zones = ["East", "North", "South", "West"] if split == "development" else ["Urban", "Coastal", "Mountain", "Rural"]
    zone = rng.choice(zones[1:])
    body = (_button("e1", "Shipping settings") + _button("e2", "Billing settings")
            + '<section id="shipping" hidden><h2>Shipping</h2><label>Delivery zone <select data-id="e3" aria-label="Delivery zone">'
            + ''.join(f'<option value="{z}">{z}</option>' for z in zones)
            + '</select></label>' + _button("e4", "Save shipping") + '</section>'
              '<section id="billing" hidden><h2>Billing</h2><p>No billing changes are needed.</p></section>')
    script = ("byId('e1').onclick=()=>{document.querySelector('#shipping').hidden=false;document.querySelector('#billing').hidden=true;};"
              "byId('e2').onclick=()=>{document.querySelector('#shipping').hidden=true;document.querySelector('#billing').hidden=false;};"
              "byId('e4').onclick=()=>finish(byId('e3').value===" + json.dumps(zone) + ");")
    tasks.append(_task(split, "shipping", f"Set the shipping delivery zone to {zone} and save it.",
                       "Account settings", body, script, ["e1", "e2", "e3", "e4"],
                       [_action("click", "e1"), _action("select", "e3", value=zone), _action("click", "e4")]))

    # Radio selection plus a second confirmation page.
    times = ["09:15", "11:45", "14:30"] if split == "development" else ["08:30", "12:15", "16:45"]
    rng.shuffle(times)
    time_index = rng.randrange(len(times))
    desired_time = times[time_index]
    body = '<p>Choose an appointment time, then confirm your booking.</p>'
    for n, slot in enumerate(times, 1):
        body += f'<label><input type="radio" name="slot" value="{slot}" data-id="e{n}"> {slot}</label>'
    body += _button("e4", "Continue") + '<section id="confirmation" hidden><p id="summary"></p>' + _button("e5", "Confirm booking") + '</section>'
    script = ("let chosen=null;const bookings=[];byId('e4').onclick=()=>{const selected=document.querySelector('input[name=slot]:checked');if(selected){chosen=selected.value;document.querySelector('#summary').textContent='Booking at '+chosen;document.querySelector('#confirmation').hidden=false;}};"
              "byId('e5').onclick=()=>{bookings.push(chosen);finish(bookings.length===1&&bookings[0]===" + json.dumps(desired_time) + ");};")
    tasks.append(_task(split, "appointment", f"Book an appointment for {desired_time} and confirm the booking.",
                       "Appointment booking", body, script, ["e1", "e2", "e3", "e4", "e5"],
                       [_action("click", f"e{time_index+1}"), _action("click", "e4"), _action("click", "e5")]))
    return tuple(tasks)
