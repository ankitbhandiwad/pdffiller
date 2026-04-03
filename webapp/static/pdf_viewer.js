(() => {
  const config = window.PDF_VIEWER_CONFIG || {};
  const fileId = config.fileId;
  if (!fileId) return;
  if (config.mode === "fill") {
    document.body.classList.add("viewer-fill-mode");
  }

  const image = document.getElementById("pdf-image");
  const boxesLayer = document.getElementById("viewer-boxes");
  const inputsLayer = document.getElementById("viewer-inputs");
  const overlay = document.getElementById("viewer-overlay");
  const selection = document.getElementById("viewer-selection");
  const wrap = document.getElementById("viewer-wrap");
  const stage = document.getElementById("viewer-stage");
  const addToggle = document.getElementById("add-toggle");
  const addCheckbox = document.getElementById("add-checkbox");
  const labelInput = document.getElementById("field-label");
  const removeBtn = document.getElementById("remove-btn");
  const statusEl = document.getElementById("viewer-status");
  const pageInput = document.getElementById("page-input");
  const pageTotal = document.getElementById("page-total");
  const prevPage = document.getElementById("prev-page");
  const nextPage = document.getElementById("next-page");
  const zoomIn = document.getElementById("zoom-in");
  const zoomOut = document.getElementById("zoom-out");

  let pageCount = 1;
  let pageIndex = 0;
  let scale = 1;
  let autoScale = true;
  let addMode = false;
  let addKind = "text";
  let dragStart = null;
  let removeMode = false;
  let lastImageWidth = 0;
  let lastImageHeight = 0;
  let pageSizes = [];
  let fieldSchema = [];
  let fieldValues = new Map();

  function setStatus(message) {
    if (statusEl) statusEl.textContent = message;
  }

  function setAddMode(on, kind) {
    if (kind) {
      addKind = kind;
    }
    if (on && removeMode) {
      setRemoveMode(false);
    }
    addMode = on;
    const textActive = addMode && addKind === "text";
    const checkboxActive = addMode && addKind === "checkbox";
    if (addToggle) {
      addToggle.classList.toggle("active", textActive);
      addToggle.textContent = textActive ? "Cancel textbox" : "Add textbox";
    }
    if (addCheckbox) {
      addCheckbox.classList.toggle("active", checkboxActive);
      addCheckbox.textContent = checkboxActive ? "Cancel checkbox" : "Add checkbox";
    }
    if (overlay) {
      overlay.classList.toggle("active", on);
    }
    if (inputsLayer) {
      inputsLayer.classList.toggle("disabled", on);
    }
    if (selection) {
      selection.classList.add("hidden");
    }
    if (on) {
      const label = addKind === "checkbox" ? "checkbox" : "textbox";
      setStatus(`Drag on the page to add a ${label}.`);
    } else {
      setStatus("Ready.");
    }
  }

  function setRemoveMode(on) {
    removeMode = on;
    if (on && addMode) {
      setAddMode(false);
    }
    if (removeBtn) {
      removeBtn.classList.toggle("active", on);
      removeBtn.textContent = on ? "Cancel remove" : "Remove";
    }
    if (boxesLayer) {
      boxesLayer.classList.toggle("remove-active", on);
    }
    if (inputsLayer) {
      inputsLayer.classList.toggle("disabled", on);
    }
    if (on) {
      setStatus("Click a box to remove.");
    } else if (!addMode) {
      setStatus("Ready.");
    }
  }

  async function loadInfo() {
    const res = await fetch(`/pdf-info/${fileId}`);
    if (!res.ok) return;
    const data = await res.json();
    pageCount = Math.max(1, Number(data.page_count || 1));
    pageSizes = Array.isArray(data.page_sizes) ? data.page_sizes : [];
    if (pageInput) pageInput.max = `${pageCount}`;
    if (pageTotal) pageTotal.textContent = `/ ${pageCount}`;
  }

  function applyScale() {
    if (!image || !wrap || !overlay) return;
    const width = Math.max(1, Math.round(lastImageWidth * scale));
    const height = Math.max(1, Math.round(lastImageHeight * scale));
    image.style.width = `${width}px`;
    image.style.height = `${height}px`;
    wrap.style.width = `${width}px`;
    wrap.style.height = `${height}px`;
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;
    if (boxesLayer) {
      boxesLayer.style.width = `${width}px`;
      boxesLayer.style.height = `${height}px`;
    }
    if (inputsLayer) {
      inputsLayer.style.width = `${width}px`;
      inputsLayer.style.height = `${height}px`;
    }
    renderFieldBoxes();
    renderFieldInputs();
  }

  function computeFitScale() {
    if (!stage || !lastImageWidth || !lastImageHeight) return 1;
    const maxW = Math.max(1, stage.clientWidth - 24);
    const scaleW = maxW / lastImageWidth;
    const fit = Math.min(scaleW, 2.5);
    return Math.max(0.2, fit);
  }

  async function loadPage() {
    if (!image) return;
    setStatus(`Loading page ${pageIndex + 1}...`);
    image.src = `/page-image/${fileId}/${pageIndex}?v=${Date.now()}`;
  }

  function notifyFields(data) {
    const api = window.PDF_VIEWER_API;
    if (api && typeof api.onFieldsUpdate === "function") {
      api.onFieldsUpdate(data);
    }
  }

  function notifyInputsRendered() {
    const api = window.PDF_VIEWER_API;
    if (api && typeof api.onInputsRendered === "function") {
      api.onInputsRendered();
    }
  }

  function applyFields(data) {
    fieldSchema = Array.isArray(data.field_schema) ? data.field_schema : [];
    const fields = data.fields || [];
    if (removeBtn) removeBtn.disabled = fields.length === 0;
    renderFieldBoxes();
    renderFieldInputs();
    notifyFields(data);
    if (!addMode) {
      if (fields.length === 0) {
        setStatus("No fields detected.");
      } else {
        setStatus(`Loaded ${fields.length} fields.`);
      }
    }
  }

  async function loadFields() {
    const res = await fetch(`/fields/${fileId}`);
    if (!res.ok) return;
    const data = await res.json();
    applyFields(data);
  }

  async function addFieldBox(coords) {
    setStatus("Adding textbox…");
    const label = labelInput ? labelInput.value.trim() : "";
    const res = await fetch(`/fields/${fileId}/add`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        page_index: pageIndex,
        x0: coords.x0,
        y0: coords.y0,
        x1: coords.x1,
        y1: coords.y1,
        label: label || null,
        field_type: addKind,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      setStatus(data.error || "Failed to add field.");
      return;
    }
    applyFields(data);
    setStatus("Textbox added.");
  }

  async function removeFieldByName(name, rect, pageIndexOverride) {
    if (!name) return;
    setStatus("Removing field…");
    const payload = { names: [name] };
    if (Array.isArray(rect) && rect.length === 4) {
      payload.rect = rect;
      payload.page_index = Number.isFinite(pageIndexOverride)
        ? pageIndexOverride
        : pageIndex;
    }
    const res = await fetch(`/fields/${fileId}/remove`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      setStatus(data.error || "Failed to remove field.");
      return;
    }
    applyFields(data);
    setStatus("Field removed.");
  }

  function renderFieldBoxes() {
    if (!boxesLayer) {
      setStatus("Field overlay unavailable.");
      return;
    }
    if (!lastImageWidth || !lastImageHeight) return;
    boxesLayer.innerHTML = "";
    const pageSize = pageSizes[pageIndex];
    if (!pageSize) {
      setStatus("Missing page size metadata.");
      return;
    }
    if (!fieldSchema.length) return;
    const pageW = Number(pageSize.width || 0);
    const pageH = Number(pageSize.height || 0);
    if (!pageW || !pageH) return;
    const fields = fieldSchema.filter((f) => f.page_index === pageIndex);
    fields.forEach((field) => {
      const rect = field.rect || [];
      if (!Array.isArray(rect) || rect.length !== 4) return;
      const x0 = Number(rect[0]);
      const y0 = Number(rect[1]);
      const x1 = Number(rect[2]);
      const y1 = Number(rect[3]);
      const left = (x0 / pageW) * lastImageWidth * scale;
      const right = (x1 / pageW) * lastImageWidth * scale;
      const top = ((pageH - y1) / pageH) * lastImageHeight * scale;
      const bottom = ((pageH - y0) / pageH) * lastImageHeight * scale;
      const width = Math.max(1, right - left);
      const height = Math.max(1, bottom - top);
      const box = document.createElement("div");
      box.className = "viewer-box";
      box.dataset.fieldName = field.name || "";
      box.dataset.fieldLabel = field.label || "";
      box.dataset.pageIndex = `${field.page_index ?? pageIndex}`;
      if (Array.isArray(field.rect) && field.rect.length === 4) {
        box.dataset.rect = field.rect.join(",");
      }
      box.style.left = `${left}px`;
      box.style.top = `${top}px`;
      box.style.width = `${width}px`;
      box.style.height = `${height}px`;
      box.title = field.label || field.name || "Field";
      boxesLayer.appendChild(box);
    });
  }

  function isCheckboxField(field) {
    const type = String(field.field_type || "").toLowerCase();
    if (type === "checkbox" || type === "btn" || type === "/btn") {
      return true;
    }
    const name = String(field.name || "").toLowerCase();
    const label = String(field.label || "").toLowerCase();
    return (
      name.startsWith("choicebutton") ||
      name.includes("checkbox") ||
      label.includes("checkbox")
    );
  }

  function snapshotInputValues() {
    if (!inputsLayer) return;
    const inputs = inputsLayer.querySelectorAll("input.viewer-input");
    inputs.forEach((input) => {
      const name = input.name;
      if (!name) return;
      if (input.type === "checkbox") {
        fieldValues.set(name, input.checked ? "Yes" : "");
      } else {
        fieldValues.set(name, input.value ?? "");
      }
    });
  }

  function renderFieldInputs() {
    if (!inputsLayer) return;
    if (!lastImageWidth || !lastImageHeight) return;
    const pageSize = pageSizes[pageIndex];
    if (!pageSize) return;
    snapshotInputValues();
    inputsLayer.innerHTML = "";
    const pageW = Number(pageSize.width || 0);
    const pageH = Number(pageSize.height || 0);
    if (!pageW || !pageH) return;
    const fields = fieldSchema.filter((f) => f.page_index === pageIndex);
    fields.forEach((field) => {
      const rect = field.rect || [];
      if (!Array.isArray(rect) || rect.length !== 4) return;
      const x0 = Number(rect[0]);
      const y0 = Number(rect[1]);
      const x1 = Number(rect[2]);
      const y1 = Number(rect[3]);
      const left = (x0 / pageW) * lastImageWidth * scale;
      const right = (x1 / pageW) * lastImageWidth * scale;
      const top = ((pageH - y1) / pageH) * lastImageHeight * scale;
      const bottom = ((pageH - y0) / pageH) * lastImageHeight * scale;
      const width = Math.max(1, right - left);
      const height = Math.max(1, bottom - top);
      const input = document.createElement("input");
      input.className = "viewer-input";
      input.type = isCheckboxField(field) ? "checkbox" : "text";
      if (input.type === "checkbox") {
        input.classList.add("checkbox");
        input.checked = fieldValues.get(field.name) === "Yes";
      } else {
        input.value = fieldValues.get(field.name) || "";
      }
      input.name = field.name || "";
      input.style.left = `${left}px`;
      input.style.top = `${top}px`;
      input.style.width = `${width}px`;
      input.style.height = `${height}px`;
      input.addEventListener("input", () => {
        if (!input.name) return;
        fieldValues.set(input.name, input.value ?? "");
      });
      input.addEventListener("change", () => {
        if (!input.name) return;
        if (input.type === "checkbox") {
          fieldValues.set(input.name, input.checked ? "Yes" : "");
        }
      });
      inputsLayer.appendChild(input);
    });
    notifyInputsRendered();
  }

  function handleBoxClick(event) {
    if (!removeMode) return;
    const target = event.target;
    if (!target) return;
    const box = target.closest(".viewer-box");
    if (!box || !box.dataset) return;
    const name = box.dataset.fieldName;
    if (!name) return;
    const rectRaw = box.dataset.rect;
    const rect = rectRaw
      ? rectRaw.split(",").map((value) => Number(value))
      : null;
    const page = Number(box.dataset.pageIndex ?? pageIndex);
    removeFieldByName(name, rect, page);
  }

  function handlePointerDown(event) {
    if (!addMode || !overlay) return;
    const rect = overlay.getBoundingClientRect();
    dragStart = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
    overlay.setPointerCapture(event.pointerId);
    if (selection) {
      selection.classList.remove("hidden");
      selection.style.left = `${dragStart.x}px`;
      selection.style.top = `${dragStart.y}px`;
      selection.style.width = "0px";
      selection.style.height = "0px";
    }
  }

  function handlePointerMove(event) {
    if (!dragStart || !selection || !overlay) return;
    const rect = overlay.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const left = Math.min(dragStart.x, x);
    const top = Math.min(dragStart.y, y);
    const width = Math.abs(dragStart.x - x);
    const height = Math.abs(dragStart.y - y);
    selection.style.left = `${left}px`;
    selection.style.top = `${top}px`;
    selection.style.width = `${width}px`;
    selection.style.height = `${height}px`;
  }

  function handlePointerUp(event) {
    if (!dragStart || !overlay) return;
    const rect = overlay.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const left = Math.min(dragStart.x, x);
    const top = Math.min(dragStart.y, y);
    const right = Math.max(dragStart.x, x);
    const bottom = Math.max(dragStart.y, y);
    const width = right - left;
    const height = bottom - top;
    dragStart = null;
    overlay.releasePointerCapture(event.pointerId);
    if (selection) {
      selection.classList.add("hidden");
    }
    if (width < 8 || height < 8) {
      setStatus("Selection too small.");
      return;
    }
    const x0 = left / rect.width;
    const y0 = top / rect.height;
    const x1 = right / rect.width;
    const y1 = bottom / rect.height;
    addFieldBox({
      x0: Math.max(0, Math.min(1, x0)),
      y0: Math.max(0, Math.min(1, y0)),
      x1: Math.max(0, Math.min(1, x1)),
      y1: Math.max(0, Math.min(1, y1)),
    });
  }

  if (addToggle) {
    addToggle.addEventListener("click", () => {
      const nextOn = !(addMode && addKind === "text");
      setAddMode(nextOn, "text");
    });
  }
  if (addCheckbox) {
    addCheckbox.addEventListener("click", () => {
      const nextOn = !(addMode && addKind === "checkbox");
      setAddMode(nextOn, "checkbox");
    });
  }
  if (removeBtn) {
    removeBtn.addEventListener("click", () => setRemoveMode(!removeMode));
  }
  if (boxesLayer) {
    boxesLayer.addEventListener("click", handleBoxClick);
  }
  if (overlay) {
    overlay.addEventListener("pointerdown", handlePointerDown);
    overlay.addEventListener("pointermove", handlePointerMove);
    overlay.addEventListener("pointerup", handlePointerUp);
  }
  if (prevPage) {
    prevPage.addEventListener("click", async () => {
      if (pageIndex <= 0) return;
      pageIndex -= 1;
      await loadPage();
    });
  }
  if (nextPage) {
    nextPage.addEventListener("click", async () => {
      if (pageIndex >= pageCount - 1) return;
      pageIndex += 1;
      await loadPage();
    });
  }
  if (pageInput) {
    pageInput.addEventListener("change", async () => {
      const value = Math.max(1, Math.min(pageCount, Number(pageInput.value)));
      pageIndex = value - 1;
      await loadPage();
    });
  }
  if (zoomIn) {
    zoomIn.addEventListener("click", () => {
      autoScale = false;
      scale = Math.min(scale + 0.25, 3);
      applyScale();
    });
  }
  if (zoomOut) {
    zoomOut.addEventListener("click", () => {
      autoScale = false;
      scale = Math.max(scale - 0.25, 0.5);
      applyScale();
    });
  }
  window.addEventListener("resize", () => {
    if (!autoScale || !lastImageWidth) return;
    scale = computeFitScale();
    applyScale();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && addMode) {
      setAddMode(false);
    }
  });

  if (image) {
    image.addEventListener("load", () => {
      lastImageWidth = image.naturalWidth || image.width;
      lastImageHeight = image.naturalHeight || image.height;
      if (autoScale) {
        scale = computeFitScale();
      }
      applyScale();
      if (pageInput) pageInput.value = `${pageIndex + 1}`;
      if (stage) {
        stage.scrollTop = 0;
        stage.scrollLeft = 0;
      }
      setStatus("Ready.");
    });
    image.addEventListener("error", () => {
      setStatus("Failed to load page image.");
    });
  }

  window.PDF_VIEWER_API = window.PDF_VIEWER_API || {};
  window.PDF_VIEWER_API.refresh = () => {
    if (autoScale && lastImageWidth) {
      scale = computeFitScale();
      applyScale();
    }
  };
  window.PDF_VIEWER_API.getFieldValues = () => {
    if (inputsLayer) {
      snapshotInputValues();
    }
    const values = {};
    fieldValues.forEach((value, key) => {
      if (value === "" || value == null) return;
      values[key] = value;
    });
    return values;
  };

  loadInfo()
    .then(loadPage)
    .then(loadFields)
    .catch(() => {
      setStatus("Failed to load PDF.");
    });

  // Legacy dead/stale code moved here for reference.
  //
  // const removeSelect = document.getElementById("remove-select");
  //
  // if (removeSelect) {
  //   removeSelect.disabled = on;
  // }
  //
  // if (removeSelect) {
  //   removeSelect.innerHTML = "";
  //   fields.forEach(([name, label]) => {
  //     const option = document.createElement("option");
  //     option.value = name;
  //     option.textContent = label || name;
  //     removeSelect.appendChild(option);
  //   });
  //   removeSelect.disabled = fields.length === 0 || removeMode;
  //   if (removeBtn) removeBtn.disabled = fields.length === 0;
  // }
  //
  // async function removeField() {
  //   if (!removeSelect || !removeSelect.value) return;
  //   await removeFieldByName(removeSelect.value);
  // }
})();
