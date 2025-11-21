function askAI() {
    let msg = document.getElementById("message").value;
    if (!msg) return;

    let history = document.getElementById("history");
    history.innerHTML += "<div>You: " + msg + "</div>";

    fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + OPENAI_KEY
        },
        body: JSON.stringify({
            model: "gpt-3.5-turbo",
            messages: [{ role: "user", content: msg }]
        })
    })
    .then(r => r.json())
    .then(d => {
        let reply = d.choices[0].message.content;
        history.innerHTML += "<div>AI: " + reply + "</div>";
    });

    document.getElementById("message").value = "";
}
