// Synapse Memory Background Service Worker

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "synapse-remember-selection",
    title: "🧠 Save to Synapse Memory",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "synapse-remember-selection" && info.selectionText) {
    const text = info.selectionText.trim();
    const source = tab?.url || "web-browser";

    try {
      const response = await fetch("http://127.0.0.1:8484/api/remember", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: `[Source: ${source}]\n${text}`,
          category: "web",
          tags: ["web-clipper", "browser"],
          importance: 0.6
        })
      });

      if (response.ok) {
        console.log("Memory saved to Synapse successfully!");
      } else {
        console.error("Failed to save memory to Synapse:", await response.text());
      }
    } catch (err) {
      console.error("Could not reach local Synapse server at http://127.0.0.1:8484. Is `synapse ui` running?", err);
    }
  }
});
