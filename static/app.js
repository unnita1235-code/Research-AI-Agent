/* global marked, DOMPurify */
(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const err = $("#err");
  const log = $("#log");
  const out = $("#out");
  const outSkel = $("#out-skeleton");
  const go = $("#go");
  const jobbar = $("#jobbar");
  const jid = $("#jid");
  const jobtime = $("#jobtime");
  const statusPill = $("#status-pill");
  const reportToolbar = $("#report-toolbar");
  const topicInput = $("#topic");
  const chips = $("#chips");
  const settingsStrip = $("#settings-strip");
  const headerBadges = $("#header-badges");
  const copyJobId = $("#copy-job-id");
  const verFoot = $("#ver-foot");
  const execBlurb = $("#exec-blurb");
  const aiExtraBlurb = $("#ai-extra-blurb");
  const askWrap = $("#ask-wrap");
  const askQ = $("#ask-q");
  const askAns = $("#ask-ans");
  const jobHistory = $("#job-history");
  const jobHistoryList = $("#job-history-list");
  const logWrap = $("#log-wrap");
  const stepEls = () => document.querySelectorAll("#stepper .step");

  let appConfig = null;
  let latestMd = "";
  let currentJobId = null;
  let jobStartMs = 0;
  let timeInterval = null;
  /** Set true when SSE receives `done` (success or fail) to avoid onerror re-poll noise. */
  let streamJobFinished = false;

  const STEP_LABELS = {
    "Query Generator": 0,
    "Tavily Searcher": 1,
    "Tavily Searcher (wave 2)": 1,
    "Fact Extractor": 2,
    "Fact Extractor (wave 2)": 2,
    "Gap query planner": 0,
    "Report Writer": 3,
  };

  function relTime(ms) {
    const s = Math.floor(ms / 1000);
    if (s < 60) return s + "s";
    return Math.floor(s / 60) + "m " + (s % 60) + "s";
  }

  function setJobTimer() {
    if (timeInterval) clearInterval(timeInterval);
    if (jobtime) {
      jobtime.textContent = jobStartMs ? "· " + relTime(Date.now() - jobStartMs) : "";
    }
    if (jobStartMs && jobtime) {
      timeInterval = setInterval(function () {
        jobtime.textContent = "· " + relTime(Date.now() - jobStartMs);
      }, 2000);
    }
  }

  function setStatusPill(text) {
    if (!statusPill) return;
    statusPill.textContent = text;
    statusPill.className = "job-pill";
    const t = (text || "").toLowerCase();
    if (t === "pending") statusPill.classList.add("job-pill--pending");
    else if (t === "running" || t === "in progress")
      statusPill.classList.add("job-pill--running");
    else if (t === "completed" || t === "done")
      statusPill.classList.add("job-pill--done");
    else if (t === "failed" || t === "error")
      statusPill.classList.add("job-pill--failed");
    else statusPill.classList.add("job-pill--pending");
  }

  function resetStepper() {
    stepEls().forEach(function (el) {
      el.classList.remove("step--active", "step--complete");
    });
  }

  function setStepperFromLabel(label) {
    const idx = STEP_LABELS[label];
    if (idx === undefined) return;
    const steps = document.querySelectorAll("#stepper .step");
    for (let i = 0; i < steps.length; i++) {
      if (i < idx) {
        steps[i].classList.add("step--complete");
        steps[i].classList.remove("step--active");
      } else if (i === idx) {
        steps[i].classList.add("step--active");
        for (let j = 0; j < i; j++) steps[j].classList.add("step--complete");
      } else {
        steps[i].classList.remove("step--active", "step--complete");
      }
    }
  }

  function finishStepper() {
    stepEls().forEach(function (el) {
      el.classList.remove("step--active");
      el.classList.add("step--complete");
    });
  }

  function showError(msg) {
    err.textContent = msg;
    err.classList.remove("hidden");
  }

  function clearError() {
    err.classList.add("hidden");
    err.textContent = "";
  }

  function logLine(s, icon) {
    const p = document.createElement("p");
    if (icon) {
      const sp = document.createElement("span");
      sp.className = "log-icon";
      sp.textContent = "→";
      p.appendChild(sp);
    }
    p.appendChild(document.createTextNode(s));
    if (log.querySelector("p.muted-placeholder")) log.innerHTML = "";
    log.appendChild(p);
    log.scrollTop = log.scrollHeight;
  }

  function renderMd(md) {
    if (!md || !out) return;
    if (typeof window.marked === "undefined" || typeof window.DOMPurify === "undefined") {
      out.textContent = "(Loading libraries…)";
      return;
    }
    const raw = window.marked.parse(md, { headerIds: true });
    out.innerHTML = DOMPurify.sanitize(raw, { USE_PROFILES: { html: true } });
    if (reportToolbar) {
      reportToolbar.hidden = !md || md.length < 5;
    }
  }

  function copyToClipboard(s) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(s);
    }
    return Promise.reject();
  }

  function researchOptionsFromForm() {
    var dEl = document.getElementById("opt-depth");
    var audEl = document.getElementById("opt-audience");
    var stEl = document.getElementById("opt-style");
    var brEl = document.getElementById("opt-breadth");
    var msEl = document.getElementById("opt-maxsrc");
    var mrEl = document.getElementById("opt-maxres");
    var depth = dEl ? parseInt(dEl.value, 10) || 1 : 1;
    return {
      depth: depth,
      audience: (audEl && audEl.value) || "general",
      output_style: (stEl && stEl.value) || "bullets",
      search_breadth: Math.min(7, Math.max(3, parseInt((brEl && brEl.value) || "4", 10) || 4)),
      max_sources: Math.min(200, Math.max(3, parseInt((msEl && msEl.value) || "24", 10) || 24)),
      max_results_per_query: Math.min(10, Math.max(1, parseInt((mrEl && mrEl.value) || "5", 10) || 5)),
    };
  }

  function loadJobHistory() {
    if (!jobHistoryList || !jobHistory) return;
    fetch("/jobs?limit=12")
      .then(function (r) {
        return r.json();
      })
      .then(function (j) {
        var jobs = (j && j.jobs) || [];
        if (!jobs.length) {
          jobHistory.classList.add("hidden");
          return;
        }
        jobHistory.classList.remove("hidden");
        jobHistoryList.innerHTML = "";
        jobs.forEach(function (row) {
          var li = document.createElement("li");
          var a = document.createElement("button");
          a.type = "button";
          a.className = "text-left text-[0.8rem] hover:opacity-80";
          a.style.cssText = "color:var(--accent-bright);background:none;border:none;cursor:pointer;padding:0";
          a.textContent = (row.status || "?") + " · " + (row.topic || row.id).slice(0, 64);
          a.addEventListener("click", function () {
            resumeJob(row.id);
          });
          li.appendChild(a);
          jobHistoryList.appendChild(li);
        });
      })
      .catch(function () {
        if (jobHistory) jobHistory.classList.add("hidden");
      });
  }

  function resumeJob(id) {
    if (!id) return;
    fetch("/research/" + id)
      .then(function (r) {
        if (!r.ok) throw new Error("not found");
        return r.json();
      })
      .then(function (j) {
        currentJobId = id;
        streamJobFinished = true;
        if (topicInput && j.topic) topicInput.value = j.topic;
        if (j.report) {
          latestMd = j.report;
          renderMd(j.report);
        }
        if (jobbar) jobbar.classList.remove("hidden");
        if (jid) {
          jid.textContent = id.length > 14 ? id.slice(0, 10) + "…" : id;
          jid.setAttribute("title", id);
        }
        setStatusPill(
          (j.status || "").toLowerCase() === "completed" ? "done" : (j.status || "unknown")
        );
        if (j.status === "completed" && appConfig && appConfig.ask_enabled && askWrap) {
          askWrap.classList.remove("hidden");
        }
      })
      .catch(function () {});
  }

  function loadChips() {
    fetch("/static/topics.json")
      .then(function (r) {
        return r.json();
      })
      .then(function (j) {
        renderChips((j && j.topics) || []);
        if (appConfig && appConfig.suggest_topics_enabled) {
          return fetch("/research/suggest-topics", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ seed: null }),
          });
        }
        return null;
      })
      .then(function (r) {
        if (r && r.ok) {
          return r.json().then(function (j) {
            if (j.topics && j.topics.length) renderChips(j.topics);
          });
        }
      })
      .catch(function () {});
  }

  function renderChips(list) {
    if (!chips) return;
    chips.innerHTML = "";
    (list || []).slice(0, 8).forEach(function (t) {
      const c = document.createElement("button");
      c.type = "button";
      c.className = "chip m-0.5";
      c.setAttribute("aria-label", "Use topic: " + t);
      c.textContent = t;
      c.addEventListener("click", function () {
        topicInput.value = t;
        topicInput.focus();
      });
      chips.appendChild(c);
    });
  }

  function loadConfig() {
    return fetch("/config")
      .then(function (r) {
        return r.json();
      })
      .then(function (c) {
        appConfig = c;
        if (verFoot) verFoot.textContent = "v" + (c.app_version || "0");
        if (headerBadges) {
          headerBadges.classList.remove("hidden");
          headerBadges.innerHTML = [
            hb("LLM: " + (c.llm_provider || "?")),
            hb(
              c.tavily_configured ? "Tavily OK" : "No Tavily",
              c.tavily_configured
            ),
            hb(c.anthropic_configured ? "Claude key" : "No Claude key", c.anthropic_configured),
          ].join(" ");
        }
        if (settingsStrip) {
          var parts = [
            "Mode: " + (c.llm_provider || "—") + " ·",
            c.tavily_configured
              ? "Tavily keys OK"
              : "Tavily key missing (503 on research)",
            " · Suggest: " + (c.suggest_topics_enabled ? "on" : "off"),
            " · AI overview: " + (c.executive_summary_enabled ? "on" : "off"),
            " · Ask: " + (c.ask_enabled ? "on" : "off"),
            " · Jobs DB: " + (c.job_db_enabled ? "on" : "off"),
            " · Rate: " + (c.rate_limit_per_min || 0) + "/min",
          ];
          settingsStrip.innerHTML = parts.join(" ");
        }
        var bexec = $("#btn-executive");
        if (bexec) bexec.hidden = !c.executive_summary_enabled;
        var btl = $("#btn-tldr");
        if (btl) btl.hidden = !c.tldr_enabled;
        var bco = $("#btn-counter");
        if (bco) bco.hidden = !c.counterarguments_enabled;
        var btr = $("#btn-translate");
        if (btr) btr.hidden = !c.translate_enabled;
        if (askWrap) askWrap.classList.toggle("hidden", !c.ask_enabled);
        loadChips();
        loadJobHistory();
      })
      .catch(function () {
        if (settingsStrip) settingsStrip.textContent = "Could not load /config (is the server running?)";
        loadChips();
      });
  }

  function hb(text, ok) {
    if (ok === false) {
      return (
        '<span class="rounded px-2 py-0.5" style="font-size:0.7rem;border:1px solid rgba(248,113,113,.4);color:#fecaca">' +
        text +
        "</span>"
      );
    }
    return (
      '<span class="rounded px-2 py-0.5" style="font-size:0.7rem;border:1px solid var(--border-subtle);color:var(--text-muted)">' +
      text +
      "</span>"
    );
  }

  function initLayout() {
    loadConfig();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initLayout);
  } else {
    initLayout();
  }

  if (copyJobId) {
    copyJobId.addEventListener("click", function () {
      if (currentJobId) {
        copyToClipboard(currentJobId).catch(function () {});
      }
    });
  }

  document.getElementById("btn-copy") &&
    document.getElementById("btn-copy").addEventListener("click", function () {
      if (latestMd) copyToClipboard(latestMd).catch(function () {});
    });

  document.getElementById("btn-dl") &&
    document.getElementById("btn-dl").addEventListener("click", function () {
      if (!latestMd) return;
      const blob = new Blob([latestMd], { type: "text/markdown;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = (topicInput.value.slice(0, 40) || "report") + "-research.md";
      a.click();
      URL.revokeObjectURL(a.href);
    });

  document.getElementById("btn-print") &&
    document.getElementById("btn-print").addEventListener("click", function () {
      window.print();
    });

  document.getElementById("btn-dl-docx") &&
    document.getElementById("btn-dl-docx").addEventListener("click", function () {
      if (!currentJobId) return;
      window.location.href = "/research/" + currentJobId + "/export?format=docx";
    });

  document.getElementById("btn-dl-html") &&
    document.getElementById("btn-dl-html").addEventListener("click", function () {
      if (!currentJobId) return;
      window.location.href = "/research/" + currentJobId + "/export?format=html";
    });

  document.getElementById("btn-ask") &&
    document.getElementById("btn-ask").addEventListener("click", function () {
      if (!currentJobId || !askQ) return;
      var q = (askQ.value || "").trim();
      if (!q) return;
      var btn = document.getElementById("btn-ask");
      if (btn) btn.disabled = true;
      fetch("/research/" + currentJobId + "/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      })
        .then(function (r) {
          return r.json().then(function (j) {
            if (r.ok) return j;
            var d = j && j.detail;
            throw new Error((d && String(d)) || r.statusText);
          });
        })
        .then(function (j) {
          if (askAns) askAns.textContent = j.answer || "";
        })
        .catch(function (e) {
          if (askAns) askAns.textContent = e.message || "Failed";
        })
        .finally(function () {
          if (btn) btn.disabled = false;
        });
    });

  function postTextExtra(suffix, btnId) {
    if (!currentJobId) return;
    var btn = document.getElementById(
      btnId || (suffix.indexOf("tldr") >= 0 ? "btn-tldr" : "btn-counter")
    );
    if (btn) btn.disabled = true;
    fetch("/research/" + currentJobId + suffix, { method: "POST" })
      .then(function (r) {
        return r.json().then(function (j) {
          if (r.ok) return j;
          throw new Error((j && j.detail) || r.statusText);
        });
      })
      .then(function (j) {
        if (aiExtraBlurb) {
          aiExtraBlurb.classList.remove("hidden");
          aiExtraBlurb.textContent = (j && j.text) || "";
        }
      })
      .catch(function (e) {
        showError(e.message || "Request failed");
      })
      .finally(function () {
        if (btn) btn.disabled = false;
      });
  }

  document.getElementById("btn-tldr") &&
    document.getElementById("btn-tldr").addEventListener("click", function () {
      postTextExtra("/tldr", "btn-tldr");
    });

  document.getElementById("btn-counter") &&
    document.getElementById("btn-counter").addEventListener("click", function () {
      postTextExtra("/counterarguments", "btn-counter");
    });

  document.getElementById("btn-translate") &&
    document.getElementById("btn-translate").addEventListener("click", function () {
      if (!currentJobId) return;
      var lang = window.prompt("Target language (e.g. Spanish, French, Japanese)", "Spanish");
      if (!lang) return;
      var btn = document.getElementById("btn-translate");
      if (btn) btn.disabled = true;
      fetch("/research/" + currentJobId + "/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_lang: lang }),
      })
        .then(function (r) {
          return r.json().then(function (j) {
            if (r.ok) return j;
            throw new Error((j && j.detail) || r.statusText);
          });
        })
        .then(function (j) {
          if (aiExtraBlurb) {
            aiExtraBlurb.classList.remove("hidden");
            aiExtraBlurb.textContent = "Translation: " + ((j && j.text) || "");
          }
        })
        .catch(function (e) {
          showError(e.message || "Translation failed");
        })
        .finally(function () {
          if (btn) btn.disabled = false;
        });
    });

  document.getElementById("btn-executive") &&
    document.getElementById("btn-executive").addEventListener("click", function () {
      if (!currentJobId) return;
      if (!appConfig || !appConfig.executive_summary_enabled) return;
      const btn = document.getElementById("btn-executive");
      btn.disabled = true;
      fetch("/research/" + currentJobId + "/executive-summary", { method: "POST" })
        .then(function (r) {
          return r.json().then(function (j) {
            if (r.ok) return j;
            var d = j && j.detail;
            if (Array.isArray(d)) {
              d = d
                .map(function (x) {
                  return (x && x.msg) || JSON.stringify(x);
                })
                .join("; ");
            }
            throw new Error((d && String(d)) || r.statusText);
          });
        })
        .then(function (j) {
          execBlurb.classList.remove("hidden");
          execBlurb.textContent = "AI overview: " + (j.summary || "");
        })
        .catch(function (e) {
          showError(e.message || "Summary failed");
        })
        .finally(function () {
          btn.disabled = false;
        });
    });

  function closeEs(es) {
    try {
      es.close();
    } catch (_) {}
  }

  document.getElementById("f").addEventListener("submit", function (e) {
    e.preventDefault();
    clearError();
    execBlurb.classList.add("hidden");
    execBlurb.textContent = "";
    const topic = topicInput.value.trim();
    if (!topic) return;
    go.disabled = true;
    streamJobFinished = false;
    latestMd = "";
    resetStepper();
    if (outSkel) {
      outSkel.classList.remove("hidden");
    }
    out.innerHTML = '<p class="text-muted" style="color:var(--text-faint)">Starting job…</p>';
    if (logWrap) logWrap.open = true;
    log.innerHTML = '<p class="muted-placeholder m-0" style="color:var(--text-faint)">Connecting…</p>';
    jobbar.classList.add("hidden");
    reportToolbar.hidden = true;

    if (aiExtraBlurb) {
      aiExtraBlurb.classList.add("hidden");
      aiExtraBlurb.textContent = "";
    }
    if (askAns) askAns.textContent = "";
    if (askWrap) askWrap.classList.add("hidden");

    fetch("/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic: topic, options: researchOptionsFromForm() }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (j) {
            var d = (j && j.detail) || res.statusText;
            d = typeof d === "string" ? d : JSON.stringify(d);
            throw new Error("HTTP " + res.status + ": " + d);
          });
        }
        return res.json();
      })
      .then(function (data) {
        const jobId = data.job_id;
        currentJobId = jobId;
        jobbar.classList.remove("hidden");
        if (jobId.length > 14) {
          jid.textContent = jobId.slice(0, 10) + "…";
          jid.setAttribute("title", jobId);
        } else {
          jid.textContent = jobId;
          jid.removeAttribute("title");
        }
        setStatusPill("pending");
        jobStartMs = Date.now();
        setJobTimer();

        if (outSkel) outSkel.classList.remove("hidden");
        const es = new EventSource("/research/" + jobId + "/stream");

        es.onmessage = function (ev) {
          var payload;
          try {
            payload = JSON.parse(ev.data);
          } catch (x) {
            return;
          }
          if (payload.type === "step") {
            setStatusPill("running");
            if (outSkel) outSkel.classList.add("hidden");
            if (payload.label) setStepperFromLabel(payload.label);
            logLine(
              "Step " +
                payload.step +
                " · " +
                payload.label +
                " — q:" +
                payload.search_queries +
                " src:" +
                payload.source_results +
                " facts:" +
                payload.extracted_facts +
                " report:" +
                (payload.final_report_len || 0) +
                " chars",
              true
            );
            if (payload.final_report) {
              latestMd = payload.final_report;
              renderMd(latestMd);
            }
          }
          if (payload.type === "done") {
            streamJobFinished = true;
            if (outSkel) outSkel.classList.add("hidden");
            if (timeInterval) clearInterval(timeInterval);
            setStatusPill(payload.status === "failed" ? "failed" : "done");
            finishStepper();
            if (payload.error) {
              showError("Job failed: " + payload.error);
            } else if (payload.final_report) {
              latestMd = payload.final_report;
              renderMd(latestMd);
            }
            if (payload.error && latestMd) renderMd(latestMd);
            logLine("Finished.", true);
            go.disabled = false;
            closeEs(es);
            if (appConfig && appConfig.ask_enabled && askWrap && payload.status !== "failed")
              askWrap.classList.remove("hidden");
            loadJobHistory();
          }
        };
        es.onerror = function () {
          if (timeInterval) clearInterval(timeInterval);
          if (streamJobFinished) {
            go.disabled = false;
            try {
              es.close();
            } catch (_) {}
            return;
          }
          if (outSkel) outSkel.classList.add("hidden");
          logLine("Stream ended; checking job…", true);
          fetch("/research/" + jobId)
            .then(function (r) {
              if (!r.ok) throw new Error("poll " + r.status);
              return r.json();
            })
            .then(function (j) {
              const st = (j.status || "").toLowerCase();
              if (st === "completed" || st === "done") {
                streamJobFinished = true;
                setStatusPill("done");
                finishStepper();
                if (j.report) {
                  latestMd = j.report;
                  renderMd(j.report);
                }
              } else if (st === "failed") {
                streamJobFinished = true;
                setStatusPill("failed");
                if (j.error) showError("Job failed: " + j.error);
                if (j.report) {
                  latestMd = j.report;
                  renderMd(j.report);
                }
              } else if (j.report && (j.report.length > 0 || st === "running")) {
                if (j.report) {
                  latestMd = j.report;
                  renderMd(j.report);
                }
                if (st === "running") {
                  setStatusPill("running");
                } else {
                  setStatusPill(st || "unknown");
                }
              }
            })
            .catch(function () {})
            .finally(function () {
              go.disabled = false;
              try {
                es.close();
              } catch (_) {}
            });
        };
      })
      .catch(function (ex) {
        if (outSkel) outSkel.classList.add("hidden");
        showError(ex.message || "Request failed");
        go.disabled = false;
      });
  });

})();
