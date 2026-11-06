document.addEventListener("DOMContentLoaded", async () => {
  const dot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const btnSave = document.getElementById("btnSave");
  const quickNote = document.getElementById("quickNote");

  // Check health of local synapse server
  try {
    const res = await fetch("http://127.0.0.1:8484/api/memories?limit=1");
    if (res.ok) {
      dot.classList.add("online");
      statusText.textContent = "Connected to Synapse Engine";
    } else {
      statusText.textContent = "Server responded with error";
    }
  } catch (e) {
    statusText.textContent = "Offline (run `synapse ui` in terminal)";
  }

  btnSave.addEventListener("click", async () => {
    const text = quickNote.value.trim();
    if (!text) return;

    btnSave.disabled = true;
    btnSave.textContent = "Saving...";

    try {
      const res = await fetch("http://127.0.0.1:8484/api/remember", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: text,
          category: "web",
          tags: ["quick-note", "browser"],
          importance: 0.7
        })
      });

      if (res.ok) {
        quickNote.value = "";
        btnSave.textContent = "✔ Saved!";
        setTimeout(() => { btnSave.textContent = "Save to Synapse"; btnSave.disabled = false; }, 1500);
      } else {
        alert("Failed to save memory.");
        btnSave.disabled = false;
        btnSave.textContent = "Save to Synapse";
      }
    } catch (e) {
      alert("Could not reach Synapse server. Please start it with `synapse ui`.");
      btnSave.disabled = false;
      btnSave.textContent = "Save to Synapse";
    }
  });
});
