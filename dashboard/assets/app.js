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
        selectedCameraId: null,
        selectedEmployeeId: null,
        page: document.body.dataset.page || "overview",
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
        if (!cameraOnline(camera)) return { label: "OFFLINE", tone: "bad" };
        if (seconds > 3) return { label: "STALE", tone: "warn" };
        return { label: "ONLINE", tone: "good" };
    }

    function workerTone(status) {
        if (status === "WORKING") return "good";
        if (status === "WALKING") return "warn";
        return "bad";
    }

    function streamUrl(path) {
        if (!path) return "";
        return `${path}${path.includes("?") ? "&" : "?"}${Date.now()}`;
    }

    function getOnlineCameras() {
        return (state.data.cameras || []).filter(cameraOnline);
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
        reconcileSelection();
        render();
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
    }

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function setHtml(id, html) {
        const el = document.getElementById(id);
        if (el) el.innerHTML = html;
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
                    setText("overview-live-name", selectedCamera.id);
                    setText("overview-live-meta", selectedCamera.stream_status?.last_error || "Camera unavailable");
                    setHtml("overview-live-badge", `<span class="pill ${stateInfo.tone}">${stateInfo.label}</span>`);
                } else {
                    liveStage.innerHTML = `<img src="${streamUrl(selectedCamera.live_view)}" alt="${escapeHtml(selectedCamera.id)}">`;
                    setText("overview-live-name", selectedCamera.id);
                    setText("overview-live-meta", `${selectedCamera.person_count || 0} people detected`);
                    setHtml("overview-live-badge", `<span class="pill ${stateInfo.tone}">${stateInfo.label}</span>`);
                }
            }
        }

        setHtml(
            "overview-active-employees",
            employees.length
                ? employees.slice(0, 6).map((worker) => `
                    <div class="employee-card" data-employee-id="${escapeHtml(worker.id)}">
                        <div class="employee-meta">
                            <div class="employee-top">
                                <div>
                                    <div class="employee-name">${escapeHtml(worker.name)}</div>
                                    <div class="meta-sub">${escapeHtml(worker.dept || "Production")} - ${escapeHtml(worker.current_camera || "unknown")}</div>
                                </div>
                                <span class="pill ${workerTone(worker.status)}">${escapeHtml(worker.status || "ACTIVE")}</span>
                            </div>
                            <div class="metrics-row">
                                <div class="mini-card"><strong>${Number(worker.total_distance_ft || 0).toFixed(1)}</strong><span>ft</span></div>
                                <div class="mini-card"><strong>${Number(worker.person_conf || 0).toFixed(2)}</strong><span>conf</span></div>
                                <div class="mini-card"><strong>${fmtSec(worker.last_seen_age)}</strong><span>seen</span></div>
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
                                        <div class="camera-name">${escapeHtml(camera.id)}</div>
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
                alerts.push({ tone: "bad", text: `${camera.id} camera feed unavailable` });
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
                                <div class="meta-sub">${escapeHtml(item.dept || "Production")} - ${escapeHtml(item.current_camera || "unknown")}</div>
                            </div>
                            <span class="pill ${workerTone(item.status)}">${escapeHtml(item.status || "ACTIVE")}</span>
                        </div>
                        <div class="metrics-row">
                            <div class="mini-card"><strong>${Number(item.total_distance_ft || 0).toFixed(1)}</strong><span>ft</span></div>
                            <div class="mini-card"><strong>${Number(item.person_conf || 0).toFixed(2)}</strong><span>conf</span></div>
                            <div class="mini-card"><strong>${fmtSec(item.last_seen_age)}</strong><span>seen</span></div>
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
                    <td>${fmtSec(item.last_seen_age)}</td>
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
            setHtml("employee-detail-cameras", rowsFromMap({}));
            setHtml("employee-detail-statuses", rowsFromMap({}));
            setText("employee-recording-path", "No recording");
            const video = document.getElementById("employee-recording-video");
            if (video) {
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
        setText("employee-detail-zone", worker.zone || "Pending");
        setHtml("employee-detail-cameras", rowsFromMap(worker.camera_times_sec));
        setHtml("employee-detail-statuses", rowsFromMap(worker.status_times_sec));
        setText("employee-recording-path", worker.path_view || "No recording");

        const video = document.getElementById("employee-recording-video");
        if (video) {
            if (worker.path_view) {
                if (video.dataset.src !== worker.path_view) {
                    video.dataset.src = worker.path_view;
                    video.src = streamUrl(worker.path_view);
                    video.load();
                }
            } else {
                video.removeAttribute("src");
                video.load();
            }
        }
    }

    function renderCamerasPage() {
        const cameras = state.data.cameras || [];
        const selected = cameras.find((camera) => camera.id === state.selectedCameraId) || null;
        setHtml(
            "cameras-list",
            cameras.length
                ? cameras.map((camera) => {
                    const stateInfo = cameraState(camera);
                    const visual = cameraOnline(camera)
                        ? `<img src="${streamUrl(camera.live_view)}" alt="${escapeHtml(camera.id)}">`
                        : `<div class="camera-frame offline">${escapeHtml(camera.stream_status?.last_error || "no live frame")}</div>`;
                    return `
                        <div class="camera-card ${camera.id === state.selectedCameraId ? "active" : ""}" data-camera-id="${escapeHtml(camera.id)}">
                            <div class="camera-frame">${visual}</div>
                            <div class="camera-meta">
                                <div class="camera-top">
                                    <div>
                                        <div class="camera-name">${escapeHtml(camera.id)}</div>
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

        const stage = document.getElementById("camera-live-stage");
        if (stage) {
            if (!selected) {
                stage.innerHTML = `<div class="live-placeholder">No online camera feed</div>`;
                setText("camera-live-name", "Camera View");
                setText("camera-live-meta", "Waiting for a working camera");
                setHtml("camera-live-badge", `<span class="pill bad">OFFLINE</span>`);
            } else {
                const stateInfo = cameraState(selected);
                if (!cameraOnline(selected)) {
                    stage.innerHTML = `<div class="live-placeholder">${escapeHtml(selected.stream_status?.last_error || "no live frame")}</div>`;
                } else {
                    stage.innerHTML = `<img src="${streamUrl(selected.live_view)}" alt="${escapeHtml(selected.id)}">`;
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
                                        <div class="record-name">${escapeHtml(camera.id)}</div>
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
                    render();
                }
            }
        });
    }

    function start() {
        bindEvents();
        loadData();
        window.setInterval(loadData, 1000);
    }

    return { start, fmtSec };
})();

window.addEventListener("DOMContentLoaded", DashboardApp.start);
