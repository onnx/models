
let currentPage = 1;
const itemsPerPage = 36;

const renderCards = function (data) {
  // function renderCards(data) {
  const mainContent = document.getElementById('main-content');
  mainContent.innerHTML = '';
  const start = (currentPage - 1) * itemsPerPage;
  const end = start + itemsPerPage;
  const pageData = data.slice(start, end);
  pageData.forEach(item => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `<h3>${item.title}</h3><p>${item.description}<br>Author: ${item.author}<br>Opset: ${item.opset}</p>`;
    const downloadButton = document.createElement('div');
    downloadButton.className = 'download-button';
    downloadButton.addEventListener('click', () => window.open(item.downloadUrl, '_blank'));
    const downloadArrow = document.createElement('div');
    downloadArrow.className = 'download-arrow';
    downloadButton.appendChild(downloadArrow);
    card.appendChild(downloadButton);
    mainContent.appendChild(card);
  });
  const pageInfo = document.getElementById('page-info');
  const totalPages = Math.ceil(data.length / itemsPerPage);
  if (pageInfo)
    pageInfo.textContent = `${currentPage}/${totalPages}`;
};

document.addEventListener('DOMContentLoaded', function () {
  const authorFilterContainer = document.getElementById('author-filters');
  const taskFilterContainer = document.getElementById('task-filters');
  const opsetFilterContainer = document.getElementById('opset-filters');

  let tasksSet = new Set();  // Create a new Set object for tasks
  let filteredData = [];
  let authorsSet = new Set();
  let opsetSet = new Set();

  async function fetchData() {
    const response = await fetch('https://api.github.com/repos/aigdat/onnx-models/git/trees/main?recursive=1');
    const data = await response.json();
    const onnxFiles = data.tree.filter(item => item.path.endsWith('.onnx') && !item.path.includes('archive/'));
    const yamlFiles = data.tree.filter(item => item.path.endsWith('turnkey_stats.yaml'));
  
    // Create a promise for each YAML file fetch operation
    const fetchYamlPromises = onnxFiles.map(file => {
      const pathParts = file.path.split('/');
      const parentDir = pathParts.slice(0, -1).join('/');
      const yamlFile = yamlFiles.find(yaml => yaml.path === `${parentDir}/turnkey_stats.yaml`);
      if (yamlFile) {
        return fetch(`https://raw.githubusercontent.com/aigdat/onnx-models/main/${yamlFile.path}`)
          .then(response => response.text())
          .then(yamlText => ({ file, yamlText }))
          .catch(error => {
            console.error('Error fetching YAML:', error);
            return { file, yamlText: null };  // Ensure the structure is consistent for error cases
          });
      }
      // If no YAML file is associated, resolve to null
      return Promise.resolve({ file, yamlText: null });
    });
  
    // Wait for all the YAML fetch operations to complete
    const results = await Promise.all(fetchYamlPromises);
  
    // Process the results to create model data
    const modelData = results.map(({ file, yamlText }) => {
      let modelTitle = file.path.split('/').pop().replace('.onnx', ''); // Default to ONNX file name
      let author = 'Unknown';
      let opset = 'NA';
      let task = 'NA';
  
      if (yamlText) {
        const yamlLines = yamlText.split('\n');
        let insideOnnxModelInfo = false;
  
        for (const line of yamlLines) {
          if (line.startsWith('onnx_model_information:')) {
            insideOnnxModelInfo = true;
            continue;
          }
          if (insideOnnxModelInfo) {
            if (line.startsWith('  opset:')) {
              opset = line.split(':')[1].trim();
            }
            if (!line.startsWith('  ')) {
              insideOnnxModelInfo = false;
            }
          }
        }
  
        const modelNameLine = yamlLines.find(line => line.startsWith('model_name:'));
        const authorLine = yamlLines.find(line => line.startsWith('author:'));
        const opsetLine = yamlLines.find(line => line.startsWith('opset:'));
        const taskLine = yamlLines.find(line => line.startsWith('task:'));
  
        if (modelNameLine) {
          modelTitle = modelNameLine.split(':')[1].trim();
        }
        if (authorLine) {
          author = authorLine.split(':')[1].trim();
          authorsSet.add(author);
        }
        if (opsetLine) {
          opset = opsetLine.split(':')[1].trim();
          opsetSet.add(opset);
        }
        if (taskLine) {
          task = taskLine.split(':')[1].trim().toLowerCase()
            .replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
          tasksSet.add(task);
        }
      }
  
      return {
        title: modelTitle,
        description: `Task: ${task}`,
        author,
        opset,
        downloadUrl: `https://github.com/aigdat/onnx-models/raw/main/${file.path}`
      };
    });
  
    // Update the filter buttons in the UI
    authorsSet.forEach(author => {
      const authorFilter = document.createElement('div');
      authorFilter.className = 'filter-button';
      authorFilter.setAttribute('data-value', author);
      authorFilter.textContent = author;
      authorFilterContainer.appendChild(authorFilter);
    });
  
    tasksSet.forEach(task => {
      const taskFilter = document.createElement('div');
      taskFilter.className = 'filter-button';
      taskFilter.setAttribute('data-value', task);
      taskFilter.textContent = task;
      taskFilterContainer.appendChild(taskFilter);
    });
  
    opsetSet.forEach(opset => {
      const opsetFilter = document.createElement('div');
      opsetFilter.className = 'filter-button';
      opsetFilter.setAttribute('data-value', opset);
      opsetFilter.textContent = opset;
      opsetFilterContainer.appendChild(opsetFilter);
    });
  
    return modelData;
  }

  fetchData().then(data => {
    filteredData = data;
    renderCards(data);

    let activeAuthorFilters = new Set();
    let activeTaskFilters = new Set();
    let activeOpsetFilters = new Set();

    const filterOptions = document.querySelectorAll('.filter-button');

    filterOptions.forEach(option => {
      option.addEventListener('click', function () {
        this.classList.toggle('active');
        const filterValue = this.getAttribute('data-value');
        const parentContainer = this.parentElement.id;

        // Add or remove the filter from the appropriate set
        if (this.classList.contains('active')) {
          if (parentContainer === 'author-filters') {
            activeAuthorFilters.add(filterValue);
          } else if (parentContainer === 'task-filters') {
            activeTaskFilters.add(filterValue);
          } else if (parentContainer === 'opset-filters') {
            activeOpsetFilters.add(filterValue);
          }
        } else {
          if (parentContainer === 'author-filters') {
            activeAuthorFilters.delete(filterValue);
          } else if (parentContainer === 'task-filters') {
            activeTaskFilters.delete(filterValue);
          } else if (parentContainer === 'opset-filters') {
            activeOpsetFilters.delete(filterValue);
          } 
        }

        // Apply the filters
        filteredData = data.filter(item => {
          const authorMatches = activeAuthorFilters.size === 0 || activeAuthorFilters.has(item.author);
          const taskMatches = activeTaskFilters.size === 0 || activeTaskFilters.has(item.description.split(': ')[1]);
          const opsetMatches = activeOpsetFilters.size === 0 || activeOpsetFilters.has(item.opset);
          return authorMatches && taskMatches && opsetMatches;
        });

        currentPage = 1;
        renderCards(filteredData);
      });
    });
  });


  const prevPage = document.getElementById('prev-page');
  const nextPage = document.getElementById('next-page');
  prevPage.addEventListener('click', () => {
    if (currentPage > 1) {
      currentPage--;
      renderCards(filteredData);
    }
  });
  nextPage.addEventListener('click', () => {
    const totalPages = Math.ceil(filteredData.length / itemsPerPage);
    if (currentPage < totalPages) {
      currentPage++;
      renderCards(filteredData);
    }
  });

  const searchBar = document.getElementById('search-bar');
  searchBar.addEventListener('input', function () {
    const query = this.value.toLowerCase();
    const searchResults = filteredData.filter(item => item.title.toLowerCase().includes(query));
    renderCards(searchResults);
  });
});

module.exports = { fetchData, renderCards };
// exports = {fetchData, renderCards}