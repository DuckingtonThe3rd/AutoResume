function sendMessage() {
    var userInput = document.getElementById("user-input").value;
    appendMessage("You: " + userInput);
    
    fetchOpenAIResponse(userInput)
      .then(response => {
        appendMessage("ChatGPT: " + response);
      })
      .catch(error => {
        console.error("Error:", error);
        appendMessage("ChatGPT: Error processing message.");
      });
  
    document.getElementById("user-input").value = "";
  }
  
  function appendMessage(message) {
    var chatBox = document.getElementById("chat-box");
    var messageElement = document.createElement("div");
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
  }
  
  async function fetchOpenAIResponse(prompt) {
    const response = await fetch("https://api.openai.com/v1/engines/text-davinci-002/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR-API-KEY-HERE"
      },
      body: JSON.stringify({
        prompt: prompt,
        max_tokens: 150
      })
    });
  
    if (!response.ok) {
      throw new Error("Error fetching response from OpenAI API.");
    }
  
    const data = await response.json();
    return data.choices[0].text.trim();
  }