// app.js - all the interactivity for UPI Shield. Plain JavaScript, no libraries.

var $ = function (id) { return document.getElementById(id); };
var NS = "http://www.w3.org/2000/svg";
var state = { last: null, demos: [], mulesLoaded: false, feedLoaded: false, reportLoaded: false, seed: 1 };

// ---------- helpers ----------
function rupees(n) { return "₹" + Number(n).toLocaleString("en-IN"); }
function wait(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

function showBanner(msg) {
    var b = $("banner");
    b.textContent = msg;
    b.hidden = !msg;
}

async function api(url, options) {
    var res;
    try {
        res = await fetch(url, options);
    } catch (e) {
        throw new Error("Cannot reach the server. Is `python api.py` still running?");
    }
    if (!res.ok) {
        var detail = "Something went wrong (" + res.status + ").";
        try {
            var j = await res.json();
            if (typeof j.detail === "string") detail = j.detail;
            else if (Array.isArray(j.detail)) detail = "Please check the form: " + j.detail[0].loc.slice(-1)[0] + " " + j.detail[0].msg + ".";
        } catch (e) { }
        throw new Error(detail);
    }
    return res.json();
}

function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
}

function svg(tag, attrs) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
}

// ---------- tabs ----------
var TABS = ["check", "feed", "mules", "report", "learn"];

function showTab() {
    var name = location.hash.replace("#", "");
    if (TABS.indexOf(name) < 0) name = "check";
    for (var i = 0; i < TABS.length; i++) {
        $("screen-" + TABS[i]).hidden = TABS[i] !== name;
    }
    var links = document.querySelectorAll(".rail-links a");
    for (var j = 0; j < links.length; j++) {
        if (links[j].dataset.tab === name) links[j].setAttribute("aria-current", "page");
        else links[j].removeAttribute("aria-current");
    }
    showBanner("");
    if (name === "feed" && !state.feedLoaded) loadFeed();
    if (name === "mules" && !state.mulesLoaded) loadMules();
    if (name === "report" && !state.reportLoaded) loadReport();
    if (name === "learn") loadLearn();
}
window.addEventListener("hashchange", showTab);

// ---------- check a payment ----------
function fillForm(p) {
    $("f-sender").value = p.sender;
    $("f-receiver").value = p.receiver;
    $("f-amount").value = p.amount;
    $("f-recent").value = p.recent;
    $("f-hour").value = p.hour;
    $("hour-out").textContent = String(p.hour).padStart(2, "0") + ":00";
    $("f-device").checked = p.new_device;
    $("f-message").value = p.message;
    var radios = document.getElementsByName("txn_type");
    for (var i = 0; i < radios.length; i++) radios[i].checked = radios[i].value === p.txn_type;
}

function readForm() {
    return {
        sender: $("f-sender").value,
        receiver: $("f-receiver").value.trim(),
        amount: parseFloat($("f-amount").value),
        txn_type: document.querySelector("input[name=txn_type]:checked").value,
        hour: parseInt($("f-hour").value, 10),
        new_device: $("f-device").checked,
        recent: parseInt($("f-recent").value, 10) || 0,
        message: $("f-message").value.trim()
    };
}

function setGate(id, stateName, text) {
    $(id).dataset.state = stateName;
    $(id + "-out").textContent = text;
}

function resetResult() {
    $("result").dataset.risk = "idle";
    setGate("gate-model", "idle", "Waiting");
    setGate("gate-message", "idle", "Waiting");
    setGate("gate-mule", "idle", "Waiting");
    $("score").textContent = "–";
    $("needle").style.transform = "rotate(-90deg)";
    $("verdict-body").hidden = true;
}

// dial arcs: 0 to 0.3 safe, 0.3 to 0.7 review, 0.7 to 1 block
function arcPath(from, to) {
    var r = 82, cx = 100, cy = 100;
    function pt(v) {
        var a = Math.PI * (1 - v);
        return (cx + r * Math.cos(a)).toFixed(2) + " " + (cy - r * Math.sin(a)).toFixed(2);
    }
    return "M " + pt(from) + " A " + r + " " + r + " 0 0 1 " + pt(to);
}

function drawDial() {
    $("arc-safe").setAttribute("d", arcPath(0, 0.3));
    $("arc-review").setAttribute("d", arcPath(0.3, 0.7));
    $("arc-block").setAttribute("d", arcPath(0.7, 1));
}

var VERDICT = {
    GREEN: "Looks safe",
    YELLOW: "Needs a second look",
    RED: "Block this payment"
};

async function screen(e) {
    if (e) e.preventDefault();
    showBanner("");
    var p = readForm();
    if (!p.receiver) { showBanner("Enter a receiver ID."); $("f-receiver").focus(); return; }
    if (!(p.amount > 0)) { showBanner("Enter an amount above zero."); $("f-amount").focus(); return; }

    var btn = $("screen-btn");
    btn.disabled = true;
    btn.textContent = "Screening…";
    resetResult();
    $("verdict-title").textContent = "Screening…";
    $("verdict-sub").textContent = "Running the three checks.";
    setGate("gate-model", "checking", "Checking");

    var data;
    try {
        var got = await Promise.all([api("/api/check", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p)
        }), wait(500)]);
        data = got[0];
    } catch (err) {
        resetResult();
        $("verdict-title").textContent = "Could not screen";
        $("verdict-sub").textContent = "Fix the problem shown above and try again.";
        showBanner(err.message);
        btn.disabled = false;
        btn.textContent = "Screen payment";
        return;
    }
    state.last = { payment: p, result: data };

    // the three checks resolve one by one
    var prob = data.model.prob;
    setGate("gate-model", prob >= 0.7 ? "bad" : prob >= 0.3 ? "warn" : "ok",
        prob >= 0.7 ? "High risk, " + prob.toFixed(2) : prob >= 0.3 ? "Some risk, " + prob.toFixed(2) : "Clear, " + prob.toFixed(2));
    setGate("gate-message", "checking", "Checking");
    await wait(350);

    if (!data.message.present) setGate("gate-message", "skip", "No message");
    else if (data.message.label === "SCAM") setGate("gate-message", "bad", "Scam, " + data.message.prob.toFixed(2));
    else setGate("gate-message", "ok", "Clear, " + data.message.prob.toFixed(2));
    setGate("gate-mule", "checking", "Checking");
    await wait(350);

    if (data.mule.found) setGate("gate-mule", "bad", data.mule.account + " is a mule");
    else setGate("gate-mule", "ok", "Not a mule");
    await wait(300);

    // the verdict lands
    var res = $("result");
    res.dataset.risk = data.label;
    $("needle").style.transform = "rotate(" + (-90 + data.final * 180).toFixed(1) + "deg)";
    $("score").textContent = data.final.toFixed(2);
    $("verdict-title").textContent = VERDICT[data.label];
    $("verdict-sub").textContent = data.advice;

    $("fact-type").textContent = data.fraud_type;
    $("fact-ratio").textContent = data.ratio + "× usual (" + rupees(data.avg) + ")";
    $("fact-payee").textContent = data.is_new_payee ? "New payee" : "Paid before";

    var list = $("reasons");
    list.innerHTML = "";
    if (data.reasons.length === 0) {
        var li = el("li", "none", "No warning signs found.");
        list.appendChild(li);
    }
    for (var i = 0; i < data.reasons.length; i++) {
        var item = el("li");
        var box = el("div");
        box.appendChild(el("span", "src", data.reasons[i].source));
        box.appendChild(el("span", "txt", data.reasons[i].text));
        item.appendChild(box);
        list.appendChild(item);
    }

    $("fb-actions").hidden = false;
    $("fb-question").hidden = false;
    $("fb-done").hidden = true;
    $("verdict-body").hidden = false;

    btn.disabled = false;
    btn.textContent = "Screen payment";
}

async function sendFeedback(wasRight) {
    if (!state.last) return;
    var r = state.last.result, p = state.last.payment;
    try {
        var info = await api("/api/feedback", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sender: p.sender, receiver: p.receiver, features: r.features,
                system_said: r.label, was_right: wasRight
            })
        });
        $("fb-actions").hidden = true;
        $("fb-question").hidden = true;
        $("fb-done").hidden = false;
        $("fb-done").textContent = "Saved (" + info.count + " in total). It will be used the next time you retrain the model.";
    } catch (err) {
        showBanner(err.message);
    }
}

function makeChips() {
    var box = $("demo-chips");
    box.innerHTML = "";
    state.demos.forEach(function (d) {
        var b = el("button", "chip", d.label);
        b.type = "button";
        b.addEventListener("click", function () {
            fillForm(d.payload);
            screen();
        });
        box.appendChild(b);
    });
}

// ---------- payment feed ----------
async function loadFeed() {
    $("feed-more").disabled = true;
    try {
        var rows = await api("/api/feed?seed=" + state.seed);
        state.feedLoaded = true;
        var body = document.querySelector("#feed-table tbody");
        body.innerHTML = "";
        var tally = { caught: 0, missed: 0, "false alarm": 0, clear: 0 };
        rows.forEach(function (r) {
            tally[r.outcome] += 1;
            var tr = el("tr");
            tr.appendChild(el("td", "", r.txn_id));
            tr.appendChild(el("td", "", r.sender));
            tr.appendChild(el("td", "", r.receiver));
            tr.appendChild(el("td", "num", rupees(r.amount)));
            tr.appendChild(el("td", "", r.txn_type.replace("_", " ").toLowerCase()));
            var risk = el("td");
            risk.appendChild(el("span", "pill " + r.label, { GREEN: "Safe", YELLOW: "Review", RED: "Block" }[r.label]));
            tr.appendChild(risk);
            tr.appendChild(el("td", "num", r.score.toFixed(2)));
            tr.appendChild(el("td", "", r.actual === "fraud" ? "Fraud: " + r.fraud_type.replace(/_/g, " ") : "Normal"));
            var cls = r.outcome === "missed" ? "res-bad" : r.outcome === "false alarm" ? "res-warn" : "res-ok";
            var text = { caught: "Fraud caught", missed: "Fraud missed", "false alarm": "False alarm", clear: "Cleared" }[r.outcome];
            tr.appendChild(el("td", cls, text));
            body.appendChild(tr);
        });
        var t = $("feed-tally");
        t.innerHTML = "";
        [["Fraud caught", tally.caught], ["Fraud missed", tally.missed], ["False alarms", tally["false alarm"]]].forEach(function (x) {
            var d = el("div");
            d.appendChild(el("b", "", String(x[1])));
            d.appendChild(el("span", "muted", x[0]));
            t.appendChild(d);
        });
    } catch (err) {
        showBanner(err.message);
    }
    $("feed-more").disabled = false;
}

// ---------- mule accounts ----------
async function loadMules() {
    try {
        var list = await api("/api/mules");
        state.mulesLoaded = true;
        var pick = $("mule-pick");
        pick.innerHTML = "";
        list.forEach(function (m) {
            var o = el("option", "", m.id + " (" + m.senders + " senders)");
            o.value = m.id;
            pick.appendChild(o);
        });
        $("mule-count").textContent = list.length + " mule accounts found in the payment graph.";
        if (list.length) showMule(list[0].id);
    } catch (err) {
        showBanner(err.message);
    }
}

function node(g, kind, x, y, label, sub) {
    var shape;
    if (kind === "mule") shape = svg("rect", { x: x - 34, y: y - 34, width: 68, height: 68, rx: 16, "class": "g-mule" });
    else if (kind === "cash") shape = svg("path", { d: "M" + x + " " + (y - 22) + " L" + (x + 22) + " " + y + " L" + x + " " + (y + 22) + " L" + (x - 22) + " " + y + " Z", "class": "g-cash" });
    else shape = svg("circle", { cx: x, cy: y, r: 12, "class": "g-victim" });
    g.appendChild(shape);
    return shape;
}

async function showMule(account) {
    try {
        var d = await api("/api/mules/" + encodeURIComponent(account));
        var stats = $("mule-stats");
        stats.innerHTML = "";
        [["People who paid it", d.senders], ["Money in", rupees(d.total_in)],
        ["Money out", rupees(d.total_out)], ["Passed on", Math.round(d.pass_on * 100) + "% in " + Math.round(d.delay_min) + " min"]
        ].forEach(function (s) {
            var w = el("div");
            w.appendChild(el("dt", "", s[0]));
            w.appendChild(el("dd", "", String(s[1])));
            stats.appendChild(w);
        });
        drawGraph(d);
    } catch (err) {
        showBanner(err.message);
    }
}

function drawGraph(d) {
    var g = $("mule-graph");
    g.innerHTML = "";
    var W = 900, H = 460, mx = 450, my = 250, vx = 130, cx = 770;

    function column(items, x, kind) {
        var n = items.length, top = 80, bottom = H - 60;
        var out = [];
        for (var i = 0; i < n; i++) {
            var y = n === 1 ? my : top + (bottom - top) * i / (n - 1);
            out.push({ item: items[i], x: x, y: y });
        }
        return out;
    }
    var v = column(d.victims, vx, "victim");
    var c = column(d.cashouts, cx, "cash");

    // edges first, so nodes sit on top
    v.forEach(function (n) {
        g.appendChild(svg("path", { d: "M" + (n.x + 12) + " " + n.y + " C 300 " + n.y + ", 330 " + my + ", " + (mx - 34) + " " + my, "class": "g-edge" }));
    });
    c.forEach(function (n) {
        g.appendChild(svg("path", { d: "M" + (mx + 34) + " " + my + " C 600 " + my + ", 620 " + n.y + ", " + (n.x - 22) + " " + n.y, "class": "g-edge flow" }));
    });

    var t1 = svg("text", { x: vx, y: 36, "text-anchor": "middle", "class": "g-title" });
    t1.textContent = "People who paid in (showing " + v.length + ")";
    g.appendChild(t1);
    var t2 = svg("text", { x: mx, y: 36, "text-anchor": "middle", "class": "g-title" });
    t2.textContent = "Mule account";
    g.appendChild(t2);
    var t3 = svg("text", { x: cx, y: 36, "text-anchor": "middle", "class": "g-title" });
    t3.textContent = "Where the money went";
    g.appendChild(t3);

    v.forEach(function (n) {
        node(g, "victim", n.x, n.y);
        var t = svg("text", { x: n.x - 22, y: n.y + 4, "text-anchor": "end", "class": "g-text" });
        t.textContent = n.item.id + "  ";
        var s2 = svg("tspan", { "class": "g-sub" });
        s2.textContent = rupees(n.item.amount);
        t.appendChild(s2);
        g.appendChild(t);
    });

    node(g, "mule", mx, my);
    var mt = svg("text", { x: mx, y: my + 5, "text-anchor": "middle", "class": "g-text" });
    mt.textContent = d.id;
    g.appendChild(mt);
    var ms = svg("text", { x: mx, y: my + 56, "text-anchor": "middle", "class": "g-sub" });
    ms.textContent = rupees(d.total_in) + " in";
    g.appendChild(ms);

    c.forEach(function (n) {
        node(g, "cash", n.x, n.y);
        var t = svg("text", { x: n.x + 30, y: n.y - 2, "class": "g-text" });
        t.textContent = n.item.id;
        g.appendChild(t);
        var s = svg("text", { x: n.x + 30, y: n.y + 14, "class": "g-sub" });
        s.textContent = rupees(n.item.amount);
        g.appendChild(s);
    });

    // legend: shape + colour, so it is not colour alone
    var lx = 40, ly = H - 12;
    [["victim", "Sender"], ["mule", "Mule"], ["cash", "Cash-out"]].forEach(function (k, i) {
        var x = lx + i * 110;
        if (k[0] === "victim") g.appendChild(svg("circle", { cx: x, cy: ly - 4, r: 6, "class": "g-victim" }));
        if (k[0] === "mule") g.appendChild(svg("rect", { x: x - 6, y: ly - 10, width: 12, height: 12, rx: 3, "class": "g-mule" }));
        if (k[0] === "cash") g.appendChild(svg("path", { d: "M" + x + " " + (ly - 11) + " L" + (x + 7) + " " + (ly - 4) + " L" + x + " " + (ly + 3) + " L" + (x - 7) + " " + (ly - 4) + " Z", "class": "g-cash" }));
        var lt = svg("text", { x: x + 14, y: ly, "class": "g-sub" });
        lt.textContent = k[1];
        g.appendChild(lt);
    });

    $("mule-cap").textContent = d.id + " received money from " + d.senders + " people and passed " +
        Math.round(d.pass_on * 100) + "% of it on within about " + Math.round(d.delay_min) + " minutes.";
}

// ---------- model report ----------
async function loadReport() {
    $("report-body").innerHTML = "<p class=\"empty\">Loading runs from MLflow…</p>";
    try {
        var data = await api("/api/report");
        state.reportLoaded = true;
        var body = $("report-body");
        body.innerHTML = "";
        var order = ["Fraud model", "Scam message", "Mule accounts", "AutoML", "Retraining"];
        order.forEach(function (gname) {
            var runs = data.runs.filter(function (r) { return r.group === gname; });
            if (!runs.length) return;
            var box = el("div", "group");
            box.appendChild(el("h2", "", gname));
            var wrap = el("div", "table-wrap");
            var table = el("table", "data");
            table.innerHTML = "<thead><tr><th>Run</th><th class='num'>Precision</th><th class='num'>Recall</th><th>F1 score</th></tr></thead>";
            var tb = el("tbody");
            runs.forEach(function (r) {
                var tr = el("tr");
                tr.appendChild(el("td", "", r.name));
                tr.appendChild(el("td", "num", r.precision.toFixed(3)));
                tr.appendChild(el("td", "num", r.recall.toFixed(3)));
                var td = el("td");
                if (r.f1 === null) td.textContent = "–";
                else {
                    var bar = el("div", "bar");
                    var track = el("span", "bar-track");
                    var fill = el("span", "bar-fill");
                    fill.style.width = (r.f1 * 100) + "%";
                    track.appendChild(fill);
                    bar.appendChild(track);
                    bar.appendChild(el("b", "", r.f1.toFixed(3)));
                    td.appendChild(bar);
                }
                tr.appendChild(td);
                tb.appendChild(tr);
            });
            table.appendChild(tb);
            wrap.appendChild(table);
            box.appendChild(wrap);
            body.appendChild(box);
        });
        if (!data.runs.length) body.appendChild(el("p", "empty", "No MLflow runs found yet. Run python run_all.py first."));
        $("shap-wrap").hidden = !data.has_shap;
    } catch (err) {
        showBanner(err.message);
    }
}

// ---------- feedback and retraining ----------
async function loadLearn() {
    try {
        var info = await api("/api/feedback");
        $("fb-count").textContent = info.count;
        $("fb-empty").hidden = info.count > 0;
        $("fb-table-wrap").hidden = info.count === 0;
        var body = document.querySelector("#fb-table tbody");
        body.innerHTML = "";
        info.recent.forEach(function (r) {
            var tr = el("tr");
            tr.appendChild(el("td", "", r.time));
            tr.appendChild(el("td", "", r.sender));
            tr.appendChild(el("td", "", r.receiver));
            tr.appendChild(el("td", "num", rupees(r.amount)));
            var td = el("td");
            td.appendChild(el("span", "pill " + r.system_said, { GREEN: "Safe", YELLOW: "Review", RED: "Block" }[r.system_said]));
            tr.appendChild(td);
            tr.appendChild(el("td", "", r.is_fraud ? "It was fraud" : "It was normal"));
            body.appendChild(tr);
        });
    } catch (err) {
        showBanner(err.message);
    }
}

function metricRow(dl, label, oldV, newV) {
    var d = newV - oldV;
    var cls = Math.abs(d) < 0.0005 ? "flat" : d > 0 ? "up" : "down";
    var sign = d > 0 ? "+" : d < 0 ? "−" : "";
    var wrap = el("div");
    wrap.appendChild(el("dt", "", label));
    var dd = el("dd");
    dd.appendChild(document.createTextNode(newV.toFixed(3)));
    dd.appendChild(el("span", "delta " + cls, sign + Math.abs(d).toFixed(3)));
    dd.appendChild(document.createElement("br"));
    dd.appendChild(el("span", "was", "was " + oldV.toFixed(3)));
    wrap.appendChild(dd);
    dl.appendChild(wrap);
}

async function retrain() {
    var btn = $("retrain-btn");
    var out = $("retrain-result");
    btn.disabled = true;
    btn.textContent = "Retraining…";
    out.innerHTML = "";
    try {
        var r = await api("/api/retrain", { method: "POST" });
        if (!r.ok) {
            out.appendChild(el("div", "notice warn", r.message));
        } else {
            var note = el("div", "notice " + (r.accepted ? "ok" : "warn"),
                r.accepted ? "New model accepted and now in use." : "New model rejected. It scored clearly worse, so the old model was kept.");
            out.appendChild(note);
            var dl = el("dl", "compare");
            metricRow(dl, "Precision", r.old.precision, r.new.precision);
            metricRow(dl, "Recall", r.old.recall, r.new.recall);
            metricRow(dl, "F1 score", r.old.f1, r.new.f1);
            out.appendChild(dl);
            out.appendChild(el("p", "muted", "Your feedback payments answered correctly: " + r.old_fb_right + " of " + r.feedback_rows +
                " before, " + r.new_fb_right + " of " + r.feedback_rows + " after."));
            state.feedLoaded = false;
            state.reportLoaded = false;
        }
    } catch (err) {
        showBanner(err.message);
    }
    btn.disabled = false;
    btn.textContent = "Retrain model";
}

// ---------- start ----------
async function init() {
    drawDial();
    $("pay-form").addEventListener("submit", screen);
    $("f-hour").addEventListener("input", function () {
        $("hour-out").textContent = String($("f-hour").value).padStart(2, "0") + ":00";
    });
    $("recent-down").addEventListener("click", function () {
        $("f-recent").value = Math.max(0, (parseInt($("f-recent").value, 10) || 0) - 1);
    });
    $("recent-up").addEventListener("click", function () {
        $("f-recent").value = Math.min(100, (parseInt($("f-recent").value, 10) || 0) + 1);
    });
    $("fb-right").addEventListener("click", function () { sendFeedback(true); });
    $("fb-wrong").addEventListener("click", function () { sendFeedback(false); });
    $("feed-more").addEventListener("click", function () { state.seed += 1; loadFeed(); });
    $("mule-pick").addEventListener("change", function () { showMule($("mule-pick").value); });
    $("retrain-btn").addEventListener("click", retrain);

    try {
        var senders = await api("/api/senders");
        var sel = $("f-sender");
        senders.forEach(function (s) {
            var o = el("option", "", s.id + "  ·  usual " + rupees(s.avg));
            o.value = s.id;
            sel.appendChild(o);
        });
        state.demos = await api("/api/demos");
        makeChips();
        fillForm(state.demos[0].payload);
    } catch (err) {
        showBanner(err.message);
    }
    showTab();
}
init();