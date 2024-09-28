document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData();
    const videoFile = document.getElementById('videoFile').files[0];
    formData.append('videoFile', videoFile);

    const processingSection = document.getElementById('processing');
    const resultSection = document.getElementById('result');
    
    // Display the processing message
    processingSection.classList.remove('hidden');
    resultSection.classList.add('hidden');

    try {
        const response = await fetch('/analyze-video', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        // Remove duplicates from title suggestions
        const uniqueTitles = Array.from(new Set(data.titleSuggestions));

        // Populate the results
        document.querySelector('#textSection .output-content').textContent = data.extractedText;
        document.querySelector('#keywordsSection .output-content').textContent = data.keywords.join(', ');
        
        const titlesContainer = document.querySelector('#titlesSection .output-content');
        titlesContainer.innerHTML = '';
        uniqueTitles.forEach((title, index) => {
            const li = document.createElement('li');
            li.textContent = `${index + 1}. ${title}`;
            titlesContainer.appendChild(li);
        });

        document.querySelector('#descriptionSection .output-content').textContent = data.videoDescription;

        const tagsContainer = document.querySelector('#tagsSection .output-content');
        tagsContainer.innerHTML = '';
        data.tags.split("\n").forEach(tag => {
            const li = document.createElement('li');
            li.textContent = tag.trim().replace('-', '');
            tagsContainer.appendChild(li);
        });

        document.querySelector('#hashtagsSection .output-content').textContent = data.hashtags.replace(/-/g, '');

        // Show result section and hide processing message
        processingSection.classList.add('hidden');
        resultSection.classList.remove('hidden');
    } catch (error) {
        console.error('Error:', error);
        processingSection.innerHTML = "<h2>Error processing video. Please try again.</h2>";
    }
});
