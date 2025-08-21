document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();
    
    let formData = new FormData(this);
    let resultBox = document.getElementById("result");
    resultBox.classList.add("d-none");

    resultBox.innerHTML = "⏳ Uploading & predicting...";
    resultBox.className = "alert alert-info mt-4";

    let response = await fetch(this.action, {
        method: "POST",
        body: formData
    });

    let text = await response.text();
    resultBox.innerHTML = text;

    if (response.ok) {
        resultBox.className = "alert alert-success mt-4";
    } else {
        resultBox.className = "alert alert-danger mt-4";
    }

    resultBox.classList.remove("d-none");
});

