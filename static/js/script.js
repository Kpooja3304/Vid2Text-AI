document.getElementById("processBtn").addEventListener("click", async function () {
    const videoUrl = document.getElementById("video_url").value;
    const transcriptLang = document.getElementById("transcript_lang").value;
    const summaryLang = document.getElementById("summary_lang").value;
    const summaryFormat = document.getElementById("summary_format").value;

    if (!videoUrl) {
        alert("⚠️ Please enter a YouTube video URL.");
        return;
    }

    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("results").classList.add("hidden");

    try {
        const response = await fetch("/process", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                video_url: videoUrl,
                transcript_lang: transcriptLang,
                summary_lang: summaryLang,
                summary_format: summaryFormat
            }),
        });

        if (!response.ok) throw new Error("Failed to process video.");

        const data = await response.json();
        document.getElementById("loading").classList.add("hidden");

        if (data.error) {
            alert("❌ Error: " + data.error);
            return;
        }

        document.getElementById("transcript_en").innerText = data.transcript_en;
        document.getElementById("transcript_selected").innerText = data.transcript_selected;
        document.getElementById("summary_en").innerText = data.summary_en;
        document.getElementById("summary_selected").innerText = data.summary_selected;

        document.getElementById("results").classList.remove("hidden");

    } catch (error) {
        document.getElementById("loading").classList.add("hidden");
        alert("❌ An error occurred: " + error.message);
    }
});
