function copyText() {
    const element = document.getElementById('output-text');
    navigator.clipboard.writeText(element.innerText);
}