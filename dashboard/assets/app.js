const DashboardApp = (() => {
    const state = {
        data: {
            timestamp: "",
            uptime: 0,
            total_person_count: 0,
            cameras: [],
            stats: [],
            performance: {
                cpu_percent: 0,
                ram_percent: 0,
                gpu_memory_percent: 0,
                gpu_util_percent: 0,
            },
            runtime: {},
            acceleration: {},
        },
        registry: {
            cameras: [],
        },
        selectedCameraId: null,
        selectedEmployeeId: null,
        selectedRegistryId: null,
        formBindingId: null,
        page: document.body.dataset.page || "overview",
        cameraListSignature: "",
        selectedLiveCameraId: null,
        modalCameraId: null,
        liveRefreshBusy: {},
    };

    function fmtSec(value) {
        const seconds = Math.max(0, Number(value || 0));
        if (seconds < 60) return `${seconds.toFixed(0)}s`;
        const minutes = Math.floor(seconds / 60);
        const rem = Math.floor(seconds % 60);
        if (minutes < 60) return `${minutes}m ${rem}s`;
        const hours = Math.floor(minutes / 60);
        return `${hours}h ${minutes % 60}m`;
    }

    function pct(value) {
        return Math.max(0, Math.min(100, Number(value || 0)));
    }

    function cameraOnline(camera) {
        return Boolean(camera && camera.stream_status && camera.stream_status.has_frame);
    }

    function cameraState(camera) {
        if (!camera) return { label: "OFFLINE", tone: "bad" };
        const seconds = Number(camera.stream_status?.seconds_since_frame || 0);
        const staleAfter = Math.max(3, Number(state.data.runtime?.stale_camera_after_seconds || 30));
        if (!cameraOnline(camera)) return { label: "OFFLINE", tone: "bad" };
        if (seconds > staleAfter) return { label: "STALE", tone: "warn" };
        return { label: "ONLINE", tone: "good" };
    }

    function workerTone(status) {
        if (status === "WORKING") return "good";
        if (status === "WALKING") return "warn";
        if (status === "IDLE_SITTING" || status === "IDLE_STANDING") return "bad";
        return "bad";
    }

    function streamUrl(path) {
        if (!path) return "";
        const normalized = path.startsWith("http") || path.startsWith("/") ? path : `/${path}`;
        return `${normalized}${normalized.includes("?") ? "&" : "?"}${Date.now()}`;
    }

    function getOnlineCameras() {
        return (state.data.cameras || []).filter(cameraOnline);
    }

    function registryCamera(cameraOrId) {
        const id = typeof cameraOrId === "string" ? cameraOrId : cameraOrId?.id;
        if (!id) return null;
        return (state.registry.cameras || []).find((camera) => camera.id === id) || null;
    }

    function cameraDisplayName(camera) {
        const registryItem = registryCamera(camera);
        return registryItem?.label || camera?.label || camera?.id || "Unknown camera";
    }

    function getEmployees() {
        return state.data.stats || [];
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function rowsFromMap(dataMap) {
        const entries = Object.entries(dataMap || {});
        if (!entries.length) {
            return `<div class="row"><span>-</span><strong>0s</strong></div>`;
        }
        return entries
            .sort((a, b) => b[1] - a[1])
            .map(([label, seconds]) => `<div class="row"><span>${escapeHtml(label)}</span><strong>${fmtSec(seconds)}</strong></div>`)
            .join("");
    }

    function totalSeconds(dataMap) {
        return Object.values(dataMap || {}).reduce((total, value) => total + Number(value || 0), 0);
    }

    function workerActiveSeconds(worker) {
        const cameraTotal = totalSeconds(worker.camera_times_sec);
        const statusTotal = totalSeconds(worker.status_times_sec);
        return Math.max(cameraTotal, statusTotal);
    }

    function ensureBreakBanner() {
        const topbar = document.querySelector(".topbar");
        if (!topbar) return null;
        let banner = document.getElementById("break-banner");
        if (!banner) {
            banner = document.createElement("div");
            banner.id = "break-banner";
            banner.className = "break-banner";
            topbar.appendChild(banner);
        }
        return banner;
    }

    async function loadData() {
        try {
            const response = await fetch("/logs/live_stats.json?" + Date.now());
            if (!response.ok) {
                throw new Error("stats unavailable");
            }
            state.data = await response.json();
        } catch {
            state.data = {
                timestamp: "",
                uptime: 0,
                total_person_count: 0,
                cameras: [],
                stats: [],
                performance: {
                    cpu_percent: 0,
                    ram_percent: 0,
                    gpu_memory_percent: 0,
                    gpu_util_percent: 0,
                },
                runtime: {},
                acceleration: {},
            };
        }
        await loadRegistry();
        reconcileSelection();
        render();
    }

    async function loadRegistry() {
        try {
            const response = await fetch("/api/cameras?" + Date.now());
            if (!response.ok) throw new Error("registry unavailable");
            state.registry = await response.json();
        } catch {
            state.registry = { cameras: [] };
        }
    }

    function reconcileSelection() {
        const onlineCameras = getOnlineCameras();
        const employees = getEmployees();
        if (!onlineCameras.some((camera) => camera.id === state.selectedCameraId)) {
            state.selectedCameraId = onlineCameras.length ? onlineCameras[0].id : null;
        }
        if (!employees.some((worker) => String(worker.id) === String(state.selectedEmployeeId))) {
            state.selectedEmployeeId = employees.length ? employees[0].id : null;
        }
        if (
            state.formBindingId !== "__new__" &&
            !(state.registry.cameras || []).some((camera) => camera.id === state.selectedRegistryId)
        ) {
            state.selectedRegistryId = state.registry.cameras?.length ? state.registry.cameras[0].id : null;
        }
    }

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function setHtml(id, html) {
        const el = document.getElementById(id);
        if (el && el.innerHTML !== html) el.innerHTML = html;
    }

    function refreshLiveImage(img, path) {
        if (!img || !path) return;
        const key = `${path}:${img.dataset.liveSlot || ""}`;
        if (state.liveRefreshBusy[key]) return;
        state.liveRefreshBusy[key] = true;
        const nextUrl = streamUrl(path);
        const loader = new Image();
        loader.onload = () => {
            img.src = nextUrl;
            img.dataset.livePath = path;
            img.style.visibility = "visible";
            state.liveRefreshBusy[key] = false;
        };
        loader.onerror = () => {
            state.liveRefreshBusy[key] = false;
        };
        loader.src = nextUrl;
    }

    function refreshVisibleLiveImages() {
        const cameras = state.data.cameras || [];
        cameras.forEach((camera) => {
            if (!cameraOnline(camera)) return;
            const img = document.querySelector(`[data-live-img="${CSS.escape(camera.id)}"]`);
            if (img) refreshLiveImage(img, camera.live_view);
        });
        const selected = cameras.find((camera) => camera.id === state.selectedCameraId);
        if (cameraOnline(selected)) {
            const img = document.querySelector("[data-selected-live-img]");
            if (img) refreshLiveImage(img, selected.live_view);
        }
        const worker = getEmployees().find((item) => String(item.id) === String(state.selectedEmployeeId));
        const workerCamera = cameras.find((camera) => camera.id === worker?.current_camera);
        if (cameraOnline(workerCamera)) {
            const img = document.getElementById("employee-live-preview");
            if (img && img.style.display !== "none") refreshLiveImage(img, workerCamera.live_view);
        }
        const modalCamera = cameras.find((camera) => camera.id === state.modalCameraId);
        if (cameraOnline(modalCamera)) {
            const img = document.getElementById("camera-modal-img");
            if (img) refreshLiveImage(img, modalCamera.live_view);
        }
    }

    function getRegistryCamera() {
        return (state.registry.cameras || []).find((camera) => camera.id === state.selectedRegistryId) || null;
    }

    function fillCameraForm(camera) {
        const item = camera || {};
        const streams = item.stream_profiles || {};
        const set = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.value = value ?? "";
        };
        set("camera-form-id", item.id || "");
        set("camera-form-label", item.label || "");
        set("camera-form-zone", item.zone || "");
        set("camera-form-role", item.role || "");
        set("camera-form-main-stream", streams.main_stream || item.video_path || "");
        set("camera-form-sub-stream", streams.sub_stream || "");
        set("camera-form-username", item.username || "");
        set("camera-form-password", item.password || "");
        set("camera-form-height", item.height_ft ?? "");
        set("camera-form-notes", item.notes || "");
        const enabled = document.getElementById("camera-form-enabled");
        if (enabled) enabled.checked = Boolean(item.enabled ?? true);
    }

    function readCameraForm() {
        const read = (id) => document.getElementById(id)?.value?.trim?.() ?? "";
        return {
            id: read("camera-form-id"),
            label: read("camera-form-label"),
            zone: read("camera-form-zone"),
            role: read("camera-form-role"),
            enabled: Boolean(document.getElementById("camera-form-enabled")?.checked),
            username: read("camera-form-username"),
            password: document.getElementById("camera-form-password")?.value ?? "",
            notes: read("camera-form-notes"),
            height_ft: read("camera-form-height") ? Number(read("camera-form-height")) : null,
            stream_profiles: {
                main_stream: read("camera-form-main-stream"),
                sub_stream: read("camera-form-sub-stream"),
            },
            video_path: read("camera-form-main-stream"),
            calibration_points: {
                camera_points: [],
                map_points: [],
            },
        };
    }

    async function saveCameraForm() {
        const camera = readCameraForm();
        if (!camera.id) {
            setText("camera-form-status", "Camera ID is required.");
            return;
        }
        const cameras = [...(state.registry.cameras || [])];
        const idx = cameras.findIndex((item) => item.id === camera.id);
        if (idx >= 0) cameras[idx] = { ...cameras[idx], ...camera };
        else cameras.push(camera);

        const response = await fetch("/api/cameras", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ cameras }),
        });
        const payload = await response.json();
        state.registry = payload;
        state.selectedRegistryId = camera.id;
        setText("camera-form-status", "Camera registry saved. Runtime will pick up changes automatically within a few seconds.");
        render();
    }

    async function testCameraForm() {
        const camera = readCameraForm();
        if (!camera.id) {
            setText("camera-form-status", "Camera ID is required before testing.");
            return;
        }
        setText("camera-form-status", "Testing camera connection...");
        const response = await fetch("/api/cameras/test", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ camera, profile: "main_stream" }),
        });
        const payload = await response.json();
        if (!payload.ok) {
            setText("camera-form-status", `Test failed: ${payload.error || "unable to read frame"}`);
            setHtml("camera-test-preview", `<div class="live-placeholder">No test snapshot</div>`);
            return;
        }
        setText("camera-form-status", `Test passed: ${payload.width}x${payload.height} snapshot captured.`);
        setHtml("camera-test-preview", `<img src="${streamUrl(payload.snapshot)}" alt="camera test snapshot">`);
    }

    function renderShell() {
        const onlineCameras = getOnlineCameras();
        const employees = getEmployees();
        const runtimeLabel = document.getElementById("runtime-label");
        const clockLabel = document.getElementById("clock-label");
        const dot = document.getElementById("status-dot");
        const navEmployees = document.getElementById("nav-employees");
        const navCameras = document.getElementById("nav-cameras");
        const navPeople = document.getElementById("nav-people");
        const navUptime = document.getElementById("nav-uptime");

        if (clockLabel) {
            const timePart = (state.data.timestamp || "").split(" ")[1] || "--:--:--";
            clockLabel.textContent = timePart;
        }
        if (runtimeLabel && dot) {
            dot.className = "state-dot";
            if (!(state.data.cameras || []).length) {
                runtimeLabel.textContent = "Waiting for backend data";
                dot.classList.add("warn");
            } else if (!onlineCameras.length) {
                runtimeLabel.textContent = "No camera feeds - check RTSP";
                dot.classList.add("bad");
            } else {
                runtimeLabel.textContent = `Runtime ready - ${onlineCameras.length} feeds`;
            }
        }

        if (navEmployees) navEmployees.textContent = `${employees.length} tracked`;
        if (navCameras) navCameras.textContent = `${onlineCameras.length} online`;
        if (navPeople) navPeople.textContent = `${state.data.total_person_count || 0} detected`;
        if (navUptime) navUptime.textContent = fmtSec(state.data.uptime);

        setText("health-cameras", `${onlineCameras.length}/${(state.data.cameras || []).length}`);
        setText("health-people", String(state.data.total_person_count || 0));
        setText("health-runtime", state.data.runtime?.device || "unknown");
        setText("health-updated", (state.data.timestamp || "").split(" ")[1] || "--:--:--");

        const breakBanner = ensureBreakBanner();
        if (breakBanner) {
            if (state.data.lunch_mode && state.data.break_message) {
                breakBanner.style.display = "flex";
                breakBanner.innerHTML = `
                    <strong>${escapeHtml(state.data.break_message)}</strong>
                    <span>${state.data.total_person_count || 0} live people detected. Shift records below are historical.</span>
                `;
            } else {
                breakBanner.style.display = "none";
                breakBanner.innerHTML = "";
            }
        }
    }

    function renderOverview() {
        const cameras = state.data.cameras || [];
        const onlineCameras = getOnlineCameras();
        const employees = getEmployees();
        const selectedCamera = cameras.find((camera) => camera.id === state.selectedCameraId) || null;
        const performance = state.data.performance || {};

        setText("overview-workers", String(employees.length));
        setText("overview-cameras", String(onlineCameras.length));
        setText("overview-people", String(state.data.total_person_count || 0));
        setText("overview-uptime", fmtSec(state.data.uptime));

        const liveStage = document.getElementById("overview-live-stage");
        if (liveStage) {
            if (!selectedCamera) {
                liveStage.innerHTML = `<div class="live-placeholder">No online camera feed</div>`;
                setText("overview-live-name", "Overview Live");
                setText("overview-live-meta", "Waiting for a working camera");
                setHtml("overview-live-badge", `<span class="pill bad">OFFLINE</span>`);
            } else {
                const stateInfo = cameraState(selectedCamera);
                if (!cameraOnline(selectedCamera)) {
                    liveStage.innerHTML = `<div class="live-placeholder">${escapeHtml(selectedCamera.stream_status?.last_error || "no live frame")}</div>`;
                    setText("overview-live-name", cameraDisplayName(selectedCamera));
                    setText("overview-live-meta", selectedCamera.stream_status?.last_error || "Camera unavailable");
                    setHtml("overview-live-badge", `<span class="pill ${stateInfo.tone}">${stateInfo.label}</span>`);
                } else {
                    liveStage.innerHTML = `<img src="${streamUrl(selectedCamera.live_view)}" alt="${escapeHtml(cameraDisplayName(selectedCamera))}">`;
                    setText("overview-live-name", cameraDisplayName(selectedCamera));
                    setText("overview-live-meta", `${selectedCamera.person_count || 0} people detected`);
                    setHtml("overview-live-badge", `<span class="pill ${stateInfo.tone}">${stateInfo.label}</span>`);
                }
            }
        }

        setHtml(
            "overview-active-employees",
            state.data.lunch_mode && !(state.data.total_person_count || 0)
                ? `<div class="list-empty">Scheduled break is active. No live employees are currently detected; previous employee cards remain in the shift history.</div>`
                : employees.length
                ? employees.slice(0, 6).map((worker) => `
                    <div class="employee-card" data-employee-id="${escapeHtml(worker.id)}">
                        <div class="employee-meta">
                            <div class="employee-top">
                                <div>
                                    <div class="employee-name">${escapeHtml(worker.name)}</div>
                                    <div class="meta-sub">${escapeHtml(worker.dept || "Production")} - ${escapeHtml(worker.current_camera || "unknown")} - ${escapeHtml(worker.activity_reason || "activity pending")}</div>
                                </div>
                                <span class="pill ${workerTone(worker.status)}">${escapeHtml(worker.status || "ACTIVE")}</span>
                            </div>
                            <div class="metrics-row">
                                <div class="mini-card"><strong>${Number(worker.total_distance_ft || 0).toFixed(1)}</strong><span>ft</span></div>
                                <div class="mini-card"><strong>${Number(worker.person_conf || 0).toFixed(2)}</strong><span>conf</span></div>
                                <div class="mini-card"><strong>${fmtSec(workerActiveSeconds(worker))}</strong><span>active</span></div>
                            </div>
                        </div>
                    </div>
                `).join("")
                : `<div class="list-empty">No identified employees yet. Detection must match a configured ArUco marker with a person box.</div>`
        );

        setHtml(
            "overview-camera-summary",
            cameras.length
                ? cameras.map((camera) => {
                    const stateInfo = cameraState(camera);
                    return `
                        <div class="camera-card" data-camera-id="${escapeHtml(camera.id)}">
                            <div class="camera-meta">
                                <div class="camera-top">
                                    <div>
                                        <div class="camera-name">${escapeHtml(cameraDisplayName(camera))}</div>
                                        <div class="meta-sub">${escapeHtml(camera.stream_status?.last_error || "live frame ok")}</div>
                                    </div>
                                    <span class="pill ${stateInfo.tone}">${stateInfo.label}</span>
                                </div>
                            </div>
                        </div>
                    `;
                }).join("")
                : `<div class="list-empty">No camera data yet.</div>`
        );

        const alerts = [];
        employees.forEach((worker) => {
            if ((worker.last_seen_age || 0) > 15) {
                alerts.push({ tone: "warn", text: `${worker.name} not seen for ${fmtSec(worker.last_seen_age)}` });
            }
            if ((worker.person_conf || 0) < 0.55) {
                alerts.push({ tone: "bad", text: `${worker.name} confidence is low (${Number(worker.person_conf || 0).toFixed(2)})` });
            }
        });
        cameras.forEach((camera) => {
            if (!cameraOnline(camera)) {
                alerts.push({ tone: "bad", text: `${cameraDisplayName(camera)} camera feed unavailable` });
            }
        });
        setHtml(
            "overview-alerts",
            alerts.length
                ? alerts.slice(0, 8).map((alert) => `
                    <div class="alert-card">
                        <div class="record-meta">
                            <div class="record-top">
                                <div class="record-name">${escapeHtml(alert.text)}</div>
                                <span class="pill ${alert.tone}">${alert.tone.toUpperCase()}</span>
                            </div>
                        </div>
                    </div>
                `).join("")
                : `<div class="list-empty">No dashboard alerts right now.</div>`
        );

        renderResourceCards("overview", performance);
    }

    function renderEmployeesPage() {
        const employees = getEmployees();
        const worker = employees.find((item) => String(item.id) === String(state.selectedEmployeeId)) || null;
        const listHtml = employees.length
            ? employees.map((item) => `
                <div class="employee-card ${String(item.id) === String(state.selectedEmployeeId) ? "active" : ""}" data-employee-id="${escapeHtml(item.id)}">
                    <div class="employee-meta">
                        <div class="employee-top">
                            <div>
                                <div class="employee-name">${escapeHtml(item.name)}</div>
                                <div class="meta-sub">${escapeHtml(item.dept || "Production")} - ${escapeHtml(item.current_camera || "unknown")} - ${escapeHtml(item.activity_reason || "activity pending")}</div>
                            </div>
                            <span class="pill ${workerTone(item.status)}">${escapeHtml(item.status || "ACTIVE")}</span>
                        </div>
                        <div class="metrics-row">
                            <div class="mini-card"><strong>${Number(item.total_distance_ft || 0).toFixed(1)}</strong><span>ft</span></div>
                            <div class="mini-card"><strong>${Number(item.person_conf || 0).toFixed(2)}</strong><span>conf</span></div>
                            <div class="mini-card"><strong>${fmtSec(workerActiveSeconds(item))}</strong><span>active</span></div>
                        </div>
                    </div>
                </div>
            `).join("")
            : `<div class="list-empty">No employee records yet.</div>`;
        setHtml("employees-list", listHtml);

        const tableHtml = employees.length
            ? employees.map((item) => `
                <tr data-employee-id="${escapeHtml(item.id)}">
                    <td>${escapeHtml(item.name)}</td>
                    <td><span class="pill ${workerTone(item.status)}">${escapeHtml(item.status || "ACTIVE")}</span></td>
                    <td>${escapeHtml(item.current_camera || "-")}</td>
                    <td>${Number(item.total_distance_ft || 0).toFixed(2)} ft</td>
                    <td>${fmtSec(workerActiveSeconds(item))}</td>
                </tr>
            `).join("")
            : `<tr><td colspan="5">No employee records yet</td></tr>`;
        setHtml("employees-table", tableHtml);

        if (!worker) {
            setText("employee-detail-name", "No employee selected");
            setText("employee-detail-sub", "Select a worker from the list");
            setText("employee-detail-avatar", "-");
            setText("employee-detail-status", "-");
            setText("employee-detail-camera", "-");
            setText("employee-detail-distance", "0 ft");
            setText("employee-detail-confidence", "0.00");
            setText("employee-detail-lastseen", "0s");
            setText("employee-detail-zone", "Pending");
            setText("employee-detail-posture", "Pending");
            setText("employee-detail-activity", "Pending");
            setHtml("employee-detail-cameras", rowsFromMap({}));
            setHtml("employee-detail-statuses", rowsFromMap({}));
            setText("employee-recording-path", "No recording");
            const video = document.getElementById("employee-recording-video");
            const preview = document.getElementById("employee-live-preview");
            if (preview) {
                preview.removeAttribute("src");
                preview.style.display = "none";
                preview.style.visibility = "hidden";
            }
            if (video) {
                video.style.display = "none";
                video.removeAttribute("src");
                video.load();
            }
            return;
        }

        setText("employee-detail-name", worker.name);
        setText("employee-detail-sub", `${worker.dept || "Production"} - ${worker.current_camera || "unknown"}`);
        setText("employee-detail-avatar", worker.name.charAt(0).toUpperCase());
        setText("employee-detail-status", worker.status || "ACTIVE");
        setText("employee-detail-camera", worker.current_camera || "-");
        setText("employee-detail-distance", `${Number(worker.total_distance_ft || 0).toFixed(2)} ft`);
        setText("employee-detail-confidence", Number(worker.person_conf || 0).toFixed(2));
        setText("employee-detail-lastseen", fmtSec(worker.last_seen_age));
        setText("employee-detail-zone", worker.zone || worker.zone_id || "Unknown Zone");
        setText("employee-detail-posture", worker.posture || "unknown");
        setText("employee-detail-activity", worker.activity_reason || "activity pending");
        setHtml("employee-detail-cameras", rowsFromMap(worker.camera_times_sec));
        setHtml("employee-detail-statuses", rowsFromMap(worker.status_times_sec));
        const currentCamera = (state.data.cameras || []).find((camera) => camera.id === worker.current_camera);
        const preview = document.getElementById("employee-live-preview");
        const recordingPath = worker.proof_view || worker.path_view || "";
        setText("employee-recording-path", recordingPath ? `${recordingPath} (proof video finalizes after backend stops)` : "No recording");

        const video = document.getElementById("employee-recording-video");
        if (preview && currentCamera?.live_view && cameraOnline(currentCamera)) {
            preview.style.display = "block";
            refreshLiveImage(preview, currentCamera.live_view);
        } else if (preview) {
            preview.style.display = "none";
            preview.style.visibility = "hidden";
        }
        if (video) {
            video.style.display = "none";
            video.removeAttribute("src");
            video.load();
        }
    }

    function renderCamerasPage() {
        const cameras = state.data.cameras || [];
        const selected = cameras.find((camera) => camera.id === state.selectedCameraId) || null;
        const modal = document.getElementById("camera-modal");
        const modalCamera = cameras.find((camera) => camera.id === state.modalCameraId);
        if (modal) {
            modal.classList.toggle("open", Boolean(modalCamera));
            modal.setAttribute("aria-hidden", modalCamera ? "false" : "true");
            if (modalCamera) {
                setText("camera-modal-title", modalCamera.id);
                setText("camera-modal-meta", `${modalCamera.person_count || 0} people detected`);
                refreshLiveImage(document.getElementById("camera-modal-img"), modalCamera.live_view);
            }
        }
        const listSignature = cameras.map((camera) => camera.id).join("|");
        if (state.cameraListSignature !== listSignature) {
            state.cameraListSignature = listSignature;
            setHtml(
                "cameras-list",
                cameras.length
                    ? cameras.map((camera) => `
                        <div class="camera-card" data-camera-id="${escapeHtml(camera.id)}">
                            <div class="camera-frame">
                                <img data-live-img="${escapeHtml(camera.id)}" data-live-slot="card-${escapeHtml(camera.id)}" alt="${escapeHtml(cameraDisplayName(camera))}" style="visibility:hidden">
                                <div class="camera-frame offline" data-camera-placeholder="${escapeHtml(camera.id)}">Loading</div>
                            </div>
                            <div class="camera-meta">
                                <div class="camera-top">
                                    <div>
                                        <div class="camera-name">${escapeHtml(cameraDisplayName(camera))}</div>
                                        <div class="meta-sub" data-camera-meta="${escapeHtml(camera.id)}">Waiting</div>
                                    </div>
                                    <span class="pill bad" data-camera-pill="${escapeHtml(camera.id)}">OFFLINE</span>
                                </div>
                            </div>
                        </div>
                    `).join("")
                    : `<div class="list-empty">No camera data yet.</div>`
            );
        }

        cameras.forEach((camera) => {
            const card = document.querySelector(`[data-camera-id="${CSS.escape(camera.id)}"]`);
            const img = document.querySelector(`[data-live-img="${CSS.escape(camera.id)}"]`);
            const placeholder = document.querySelector(`[data-camera-placeholder="${CSS.escape(camera.id)}"]`);
            const meta = document.querySelector(`[data-camera-meta="${CSS.escape(camera.id)}"]`);
            const pill = document.querySelector(`[data-camera-pill="${CSS.escape(camera.id)}"]`);
            const stateInfo = cameraState(camera);
            if (card) card.classList.toggle("active", camera.id === state.selectedCameraId);
            if (meta) meta.textContent = camera.stream_status?.last_error || "live frame ok";
            if (pill) {
                pill.className = `pill ${stateInfo.tone}`;
                pill.textContent = stateInfo.label;
            }
            if (cameraOnline(camera)) {
                if (placeholder) placeholder.style.display = "none";
                if (img) {
                    img.style.display = "block";
                    refreshLiveImage(img, camera.live_view);
                }
            } else {
                if (img) img.style.display = "none";
                if (placeholder) {
                    placeholder.style.display = "flex";
                    placeholder.textContent = camera.stream_status?.last_error || "no live frame";
                }
            }
        });

        const stage = document.getElementById("camera-live-stage");
        if (stage) {
            if (!selected) {
                stage.innerHTML = `<div class="live-placeholder">No online camera feed</div>`;
                state.selectedLiveCameraId = null;
                setText("camera-live-name", "Camera View");
                setText("camera-live-meta", "Waiting for a working camera");
                setHtml("camera-live-badge", `<span class="pill bad">OFFLINE</span>`);
            } else {
                const stateInfo = cameraState(selected);
                if (!cameraOnline(selected)) {
                    stage.innerHTML = `<div class="live-placeholder">${escapeHtml(selected.stream_status?.last_error || "no live frame")}</div>`;
                    state.selectedLiveCameraId = null;
                } else {
                    if (state.selectedLiveCameraId !== selected.id || !stage.querySelector("img")) {
                        state.selectedLiveCameraId = selected.id;
                        stage.innerHTML = `<img data-selected-live-img data-live-slot="selected-${escapeHtml(selected.id)}" alt="${escapeHtml(selected.id)}" style="visibility:hidden">`;
                    }
                    refreshLiveImage(stage.querySelector("[data-selected-live-img]"), selected.live_view);
                }
                setText("camera-live-name", selected.id);
                setText("camera-live-meta", `${selected.person_count || 0} people detected`);
                setHtml("camera-live-badge", `<span class="pill ${stateInfo.tone}">${stateInfo.label}</span>`);
            }
        }

        setText("camera-status-label", selected ? cameraState(selected).label : "-");
        setText("camera-status-error", selected?.stream_status?.last_error || "-");
        setText("camera-status-people", String(selected?.person_count || 0));
        setText("camera-status-seen", selected?.stream_status?.seconds_since_frame != null ? fmtSec(selected.stream_status.seconds_since_frame) : "-");
        setText("camera-status-reconnects", String(selected?.stream_status?.reconnects || 0));
        setText("camera-status-frames", String(selected?.stream_status?.frames_read || 0));

        const registryCameras = state.registry.cameras || [];
        setText("camera-registry-meta", `${registryCameras.length} cameras in registry`);
        setHtml(
            "camera-registry-list",
            registryCameras.length
                ? registryCameras.map((camera) => {
                    const runtimeCamera = cameras.find((item) => item.id === camera.id);
                    const stateInfo = cameraState(runtimeCamera);
                    const active = camera.id === state.selectedRegistryId ? "active" : "";
                    return `
                        <div class="registry-card ${active}" data-registry-id="${escapeHtml(camera.id)}">
                            <div class="record-top">
                                <div>
                                    <div class="record-name">${escapeHtml(camera.label || camera.id)}</div>
                                    <div class="meta-sub">${escapeHtml(camera.zone || camera.role || "unassigned")}</div>
                                </div>
                                <span class="pill ${camera.enabled ? stateInfo.tone : "warn"}">${camera.enabled ? stateInfo.label : "DISABLED"}</span>
                            </div>
                        </div>
                    `;
                }).join("")
                : `<div class="list-empty">No camera registry data yet.</div>`
        );

        const selectedRegistry = getRegistryCamera();
        if (selectedRegistry && state.formBindingId !== selectedRegistry.id) {
            fillCameraForm(selectedRegistry);
            state.formBindingId = selectedRegistry.id;
        } else if (!selectedRegistry && state.formBindingId !== "__new__") {
            fillCameraForm(null);
            state.formBindingId = "__new__";
        }
    }

    function renderSystemPage() {
        const perf = state.data.performance || {};
        const runtime = state.data.runtime || {};
        const accel = state.data.acceleration || {};
        const cameras = state.data.cameras || [];
        renderResourceCards("system", perf);

        setHtml("system-runtime", `
            <div class="row"><span>Camera Profile</span><strong>${escapeHtml(runtime.camera_profile || "-")}</strong></div>
            <div class="row"><span>Inference Width</span><strong>${escapeHtml(runtime.inference_width || "-")}</strong></div>
            <div class="row"><span>Target FPS</span><strong>${escapeHtml(runtime.target_fps || "-")}</strong></div>
            <div class="row"><span>Batch Size</span><strong>${escapeHtml(runtime.batch_size || "-")}</strong></div>
            <div class="row"><span>Path Video FPS</span><strong>${escapeHtml(runtime.path_video_fps || "-")}</strong></div>
        `);
        setHtml("system-acceleration", `
            <div class="row"><span>Selected Device</span><strong>${escapeHtml(accel.selected_device || "-")}</strong></div>
            <div class="row"><span>CUDA Build</span><strong>${escapeHtml(accel.torch_cuda_version || "-")}</strong></div>
            <div class="row"><span>GPU Name</span><strong>${escapeHtml(accel.cuda_device_name || "-")}</strong></div>
            <div class="row"><span>CUDA Available</span><strong>${escapeHtml(accel.cuda_available)}</strong></div>
            <div class="row"><span>GPU Count</span><strong>${escapeHtml(accel.cuda_device_count || 0)}</strong></div>
        `);
        setHtml(
            "system-feed-health",
            cameras.length
                ? cameras.map((camera) => {
                    const stateInfo = cameraState(camera);
                    return `
                        <div class="record-card">
                            <div class="record-meta">
                                <div class="record-top">
                                    <div>
                                        <div class="record-name">${escapeHtml(cameraDisplayName(camera))}</div>
                                        <div class="meta-sub">${escapeHtml(camera.stream_status?.last_error || "live frame ok")}</div>
                                    </div>
                                    <span class="pill ${stateInfo.tone}">${stateInfo.label}</span>
                                </div>
                            </div>
                        </div>
                    `;
                }).join("")
                : `<div class="list-empty">No backend feed data yet.</div>`
        );
        setHtml("system-summary", `
            <div class="system-card"><strong>${getOnlineCameras().length}</strong><span>Cameras online</span></div>
            <div class="system-card"><strong>${cameras.length}</strong><span>Cameras configured</span></div>
            <div class="system-card"><strong>${getEmployees().length}</strong><span>Employees tracked</span></div>
            <div class="system-card"><strong>${fmtSec(state.data.uptime)}</strong><span>Runtime uptime</span></div>
        `);
    }

    function renderResourceCards(prefix, perf) {
        const values = {
            cpu: perf.cpu_percent ?? 0,
            ram: perf.ram_percent ?? 0,
            gpu: perf.gpu_util_percent ?? perf.gpu_percent ?? 0,
            vram: perf.gpu_memory_percent ?? 0,
        };
        Object.entries(values).forEach(([name, value]) => {
            setText(`${prefix}-${name}-value`, `${pct(value).toFixed(0)}%`);
            const bar = document.getElementById(`${prefix}-${name}-bar`);
            if (bar) bar.style.width = `${pct(value)}%`;
        });
    }

    function render() {
        renderShell();
        if (state.page === "overview") renderOverview();
        if (state.page === "employees") renderEmployeesPage();
        if (state.page === "cameras") renderCamerasPage();
        if (state.page === "system") renderSystemPage();
    }

    function bindEvents() {
        document.addEventListener("click", (event) => {
            const employeeTrigger = event.target.closest("[data-employee-id]");
            if (employeeTrigger) {
                state.selectedEmployeeId = employeeTrigger.getAttribute("data-employee-id");
                render();
                return;
            }

            const cameraTrigger = event.target.closest("[data-camera-id]");
            if (cameraTrigger) {
                const id = cameraTrigger.getAttribute("data-camera-id");
                const camera = (state.data.cameras || []).find((item) => item.id === id);
                if (cameraOnline(camera)) {
                    state.selectedCameraId = id;
                    if (state.page === "cameras") state.modalCameraId = id;
                    render();
                }
                return;
            }

            if (event.target.id === "camera-modal-close" || event.target.id === "camera-modal") {
                state.modalCameraId = null;
                render();
                return;
            }

            const registryTrigger = event.target.closest("[data-registry-id]");
            if (registryTrigger) {
                state.selectedRegistryId = registryTrigger.getAttribute("data-registry-id");
                state.formBindingId = null;
                render();
                return;
            }

            if (event.target.id === "camera-form-new") {
                state.selectedRegistryId = null;
                state.formBindingId = "__new__";
                fillCameraForm(null);
                setText("camera-form-status", "New camera form ready.");
                setHtml("camera-test-preview", `<div class="live-placeholder">No test snapshot</div>`);
                return;
            }

            if (event.target.id === "camera-form-save") {
                saveCameraForm();
                return;
            }

            if (event.target.id === "camera-form-test") {
                testCameraForm();
            }
        });
    }

    function start() {
        bindEvents();
        loadData();
        window.setInterval(loadData, 1000);
        window.setInterval(refreshVisibleLiveImages, 500);
    }

    return { start, fmtSec };
})();

window.addEventListener("DOMContentLoaded", DashboardApp.start);
