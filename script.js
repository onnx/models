class DataFetcher {
  constructor() {
    this.onnxRepoContentPath =
      "https://api.github.com/repos/onnx/models/git/trees/main?recursive=1";
    this.yamlRepoFilePath =
      "https://raw.githubusercontent.com/onnx/models/main";

    this.onnxFiles = [];
    this.yamlFiles = {};

    this.yamlFileDetails = [];

    this.paginationInfo = { currentPage: 1, itemsPerPage: 20, totalPages: 0 };

    this.availableFilters = {
      authorsSet: new Set(),
      opsetSet: new Set(),
      tasksSet: new Set(),
    };

    this.activeFilters = {
      activeAuthorsSet: new Set(),
      activeOpsetSet: new Set(),
      activeTasksSet: new Set(),
      searchFilter: null,
    };
  }

  async fetchOnnxYamlFilesData() {
    const response = await fetch(this.onnxRepoContentPath);
    const data = await response.json();

    data.tree.forEach((item) => {
      if (item.path.endsWith(".onnx")) {
        this.onnxFiles.push(item);
      }

      if (item.path.endsWith("turnkey_stats.yaml")) {
        this.yamlFiles[item.path] = item.path;
      }
    });

    this.paginationInfo.totalPages = Math.ceil(
      this.onnxFiles.length / this.paginationInfo.itemsPerPage
    );

    return { onnxFiles: this.onnxFiles, yamlFiles: this.yamlFiles };
  }

  async fetchYamlFileDetails(onnxFilesList, yamlFilesMap) {
    const yamlFilePromises = onnxFilesList.map((file) => {
      const pathParts = file.path.split("/");
      const parentDir = pathParts.slice(0, -1).join("/");

      const yamlFile = yamlFilesMap[`${parentDir}/turnkey_stats.yaml`];

      if (yamlFile) {
        return fetch(`${this.yamlRepoFilePath}/${yamlFile}`)
          .then((response) => response.text())
          .then((yamlText) => ({ file, yamlText }))
          .catch((error) => {
            console.error("Error fetching YAML:", error);
            return { file, yamlText: null }; // Ensure the structure is consistent for error cases
          });
      } else {
        // If no YAML file is associated, resolve to null
        // console.log("missing yaml file", file);
        return Promise.resolve({ file, yamlText: null });
      }
    });

    const yamlFilePromiseResults = await Promise.all(yamlFilePromises);
    const processedYamlFileDetails = this.processYamlFileModels(
      yamlFilePromiseResults
    );

    this.yamlFileDetails = this.yamlFileDetails.concat(
      processedYamlFileDetails
    );

    return processedYamlFileDetails;
  }

  getFilteredResults(resetToFirstPage) {
    const { activeAuthorsSet, activeTasksSet, activeOpsetSet, searchFilter } =
      this.activeFilters;

    let filteredData = this.yamlFileDetails.filter((item) => {
      const authorMatches =
        activeAuthorsSet.size === 0 || activeAuthorsSet.has(item.author);
      const taskMatches =
        activeTasksSet.size === 0 ||
        activeTasksSet.has(item.description.split(": ")[1]);
      const opsetMatches =
        activeOpsetSet.size === 0 || activeOpsetSet.has(item.opset);
      return authorMatches && taskMatches && opsetMatches;
    });

    if (searchFilter) {
      filteredData = filteredData.filter((item) =>
        item.title.toLowerCase().includes(searchFilter)
      );
    }

    if (resetToFirstPage) {
      this.paginationInfo.currentPage = 1;
      this.paginationInfo.totalPages = Math.ceil(
        filteredData.length / this.paginationInfo.itemsPerPage
      );
    }

    return filteredData;
  }

  getPaginatedFilteredResults(resetToFirstPage) {
    const filteredData = this.getFilteredResults(resetToFirstPage);

    const currentPageIndexStart =
      (this.paginationInfo.currentPage - 1) * this.paginationInfo.itemsPerPage;
    const currentPageIndexEnd =
      currentPageIndexStart + this.paginationInfo.itemsPerPage;

    return filteredData.slice(currentPageIndexStart, currentPageIndexEnd);
  }

  processYamlFileModels(yamlFileDetails) {
    return yamlFileDetails.map(({ file, yamlText }) => {
      let modelTitle = file.path.split("/").pop().replace(".onnx", ""); // Default to ONNX file name
      let author = "Unknown";
      let opset = "Not Available";
      let task = "Not Available";

      if (yamlText) {
        const yamlLines = yamlText.split("\n");
        let insideOnnxModelInfo = false;

        for (const line of yamlLines) {
          if (line.startsWith("onnx_model_information:")) {
            insideOnnxModelInfo = true;
            continue;
          }
          if (insideOnnxModelInfo) {
            if (line.startsWith("  opset:")) {
              opset = line.split(":")[1].trim();
            }
            if (!line.startsWith("  ")) {
              insideOnnxModelInfo = false;
            }
          }
        }

        const modelNameLine = yamlLines.find((line) =>
          line.startsWith("model_name:")
        );
        const authorLine = yamlLines.find((line) => line.startsWith("author:"));
        const opsetLine = yamlLines.find((line) => line.startsWith("opset:"));
        const taskLine = yamlLines.find((line) => line.startsWith("task:"));

        if (modelNameLine) {
          modelTitle = modelNameLine.split(":")[1].trim();
        }
        if (authorLine) {
          author = authorLine.split(":")[1].trim();
        }
        if (opsetLine) {
          opset = opsetLine.split(":")[1].trim();
        }
        if (taskLine) {
          task = taskLine
            .split(":")[1]
            .trim()
            .toLowerCase()
            .replace(/_/g, " ")
            .split(" ")
            .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
            .join(" ");
        }
      }

      this.availableFilters.authorsSet.add(author);
      this.availableFilters.tasksSet.add(task);
      this.availableFilters.opsetSet.add(opset);

      return {
        title: modelTitle,
        description: `Task: ${task}`,
        author,
        task,
        opset,
        downloadUrl: `https://github.com/onnx/models/raw/main/${file.path}`,
      };
    });
  }

  updateAvailableFilters(yamlFileDetails) {
    yamlFileDetails.forEach((item) => {
      this.availableFilters.authorsSet.add(item.author);
      this.availableFilters.tasksSet.add(item.task);
      this.availableFilters.opsetSet.add(item.opset);
    });
  }
}

class RenderUI {
  renderCard(cardData) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<h3>${cardData.title}</h3><p>${cardData.description}<br>Author: ${cardData.author}<br>Opset: ${cardData.opset}</p>`;

    const downloadButton = document.createElement("div");
    downloadButton.className = "download-button";

    downloadButton.addEventListener("click", () =>
      window.open(cardData.downloadUrl, "_blank")
    );

    const downloadArrow = document.createElement("i");
    downloadArrow.className = "fa-solid fa-download";
    downloadButton.appendChild(downloadArrow);
    card.appendChild(downloadButton);

    return card;
  }

  renderCards(paginatedCardsData) {
    const cardsContent = document.getElementById("cards-content");
    cardsContent.innerHTML = "";

    if (!paginatedCardsData || paginatedCardsData.length == 0) {
      cardsContent.innerHTML = "No results matching the filters";
      document.getElementById("pagination").style.visibility = "hidden";
    } else {
      document.getElementById("pagination").style.visibility = "visible";

      paginatedCardsData.forEach((item) => {
        const card = this.renderCard(item);
        cardsContent.appendChild(card);
      });
    }
  }

  renderPaginationInfo(paginationInfo) {
    const pageInfo = document.getElementById("page-info");

    if (pageInfo) {
      pageInfo.textContent = `${paginationInfo.currentPage}/${paginationInfo.totalPages}`;

      const prevPage = document.getElementById("prev-page");
      const nextPage = document.getElementById("next-page");

      if (paginationInfo.currentPage == 1) {
        prevPage.style.opacity = "50%";
        prevPage.style["pointer-events"] = "none";
        prevPage.style.cursor = "default";
      } else {
        prevPage.style.opacity = "100%";
        prevPage.style["pointer-events"] = "all";
        prevPage.style.cursor = "pointer";
      }

      if (paginationInfo.currentPage == paginationInfo.totalPages) {
        nextPage.style.opacity = "50%";
        nextPage.style["pointer-events"] = "none";
        nextPage.style.cursor = "default";
      } else {
        nextPage.style.opacity = "100%";
        nextPage.style["pointer-events"] = "all";
        nextPage.style.cursor = "pointer";
      }
    }
  }

  renderPage(dataFetcher, resetToFirstPage) {
    this.renderCards(dataFetcher.getPaginatedFilteredResults(resetToFirstPage));
    this.renderPaginationInfo(dataFetcher.paginationInfo);
    this.renderFilters(dataFetcher.availableFilters, dataFetcher.activeFilters);

    this.attachFiltersEventListeners(dataFetcher);
  }

  renderFilters(filtersData, activeFiltersData) {
    if (!filtersData || !activeFiltersData) return;

    const { authorsSet, tasksSet, opsetSet } = filtersData;
    const { activeAuthorsSet, activeTasksSet, activeOpsetSet } =
      activeFiltersData;

    const authorFilterContainer = document.getElementById("author-filters");
    const taskFilterContainer = document.getElementById("task-filters");
    const opsetFilterContainer = document.getElementById("opset-filters");

    authorFilterContainer.innerHTML = "";
    taskFilterContainer.innerHTML = "";
    opsetFilterContainer.innerHTML = "";

    authorsSet.forEach((author) => {
      const authorFilter = document.createElement("div");
      authorFilter.className = "filter-button";
      authorFilter.setAttribute("data-value", author);
      authorFilter.textContent = author;

      if (activeAuthorsSet.has(author)) {
        authorFilter.classList.add("active");
      } else {
        authorFilter.classList.remove("active");
      }

      authorFilterContainer.appendChild(authorFilter);
    });

    tasksSet.forEach((task) => {
      const taskFilter = document.createElement("div");
      taskFilter.className = "filter-button";
      taskFilter.setAttribute("data-value", task);
      taskFilter.textContent = task;

      if (activeTasksSet.has(task)) {
        taskFilter.classList.add("active");
      } else {
        taskFilter.classList.remove("active");
      }

      taskFilterContainer.appendChild(taskFilter);
    });

    opsetSet.forEach((opset) => {
      const opsetFilter = document.createElement("div");
      opsetFilter.className = "filter-button";
      opsetFilter.setAttribute("data-value", opset);
      opsetFilter.textContent = opset;

      if (activeOpsetSet.has(opset)) {
        opsetFilter.classList.add("active");
      } else {
        opsetFilter.classList.remove("active");
      }

      opsetFilterContainer.appendChild(opsetFilter);
    });
  }

  attachPaginatedEventListeners(dataFetcher) {
    const renderUIObj = this;

    // attach event listeners for pagination
    const prevPage = document.getElementById("prev-page");
    const nextPage = document.getElementById("next-page");

    prevPage.addEventListener("click", () => {
      if (dataFetcher.paginationInfo.currentPage > 1) {
        dataFetcher.paginationInfo.currentPage--;
        renderUIObj.renderPage(dataFetcher);
      }
    });

    nextPage.addEventListener("click", () => {
      if (
        dataFetcher.paginationInfo.currentPage <
        dataFetcher.paginationInfo.totalPages
      ) {
        dataFetcher.paginationInfo.currentPage++;
        renderUIObj.renderPage(dataFetcher);
      }
    });
  }

  attachSearchBarEventListeners(dataFetcher) {
    const renderUIObj = this;

    const searchBar = document.getElementById("search-bar");

    searchBar.addEventListener("input", function () {
      dataFetcher.activeFilters.searchFilter = this.value.toLowerCase();
      renderUIObj.renderPage(dataFetcher, true);
    });
  }

  attachFiltersEventListeners(dataFetcher) {
    const renderUIObj = this;

    // attach event listeners for filters
    const filterOptions = document.querySelectorAll(".filter-button");
    filterOptions.forEach((option) => {
      option.addEventListener("click", function () {
        this.classList.toggle("active");

        const filterValue = this.getAttribute("data-value");
        const parentContainer = this.parentElement.id;

        // Add or remove the filter from the appropriate set
        if (this.classList.contains("active")) {
          if (parentContainer === "author-filters") {
            dataFetcher.activeFilters.activeAuthorsSet.add(filterValue);
          } else if (parentContainer === "task-filters") {
            dataFetcher.activeFilters.activeTasksSet.add(filterValue);
          } else if (parentContainer === "opset-filters") {
            dataFetcher.activeFilters.activeOpsetSet.add(filterValue);
          }
        } else {
          if (parentContainer === "author-filters") {
            dataFetcher.activeFilters.activeAuthorsSet.delete(filterValue);
          } else if (parentContainer === "task-filters") {
            dataFetcher.activeFilters.activeTasksSet.delete(filterValue);
          } else if (parentContainer === "opset-filters") {
            dataFetcher.activeFilters.activeOpsetSet.delete(filterValue);
          }
        }

        renderUIObj.renderPage(dataFetcher, true);
      });
    });
  }
}

const loadPage = async (renderUI, dataFetcher) => {
  const { onnxFiles, yamlFiles } = await dataFetcher.fetchOnnxYamlFilesData();

  const itemsPerPage = dataFetcher.paginationInfo.itemsPerPage;
  const totalPages = dataFetcher.paginationInfo.totalPages;
  // Fetch first Page
  await dataFetcher.fetchYamlFileDetails(
    onnxFiles.slice(0, itemsPerPage),
    yamlFiles
  );

  // Render the First Page
  renderUI.renderPage(dataFetcher);

  // Fetch Rest of the files and update the page with all available filters
  (async () => {
    const loadingFilters = document.getElementById("loading-filters");
    const loadingPagination = document.getElementById("loading-pagination");
    loadingFilters.style.display = loadingPagination.style.display = "block";

    for (let currentPage = 2; currentPage < totalPages; currentPage++) {
      let currentPageIndex = (currentPage - 1) * itemsPerPage;

      await dataFetcher.fetchYamlFileDetails(
        onnxFiles.slice(currentPageIndex, currentPageIndex + itemsPerPage),
        yamlFiles
      );

      currentPageIndex += itemsPerPage;

      localStorage.setItem(
        "OnnxYamlFileDetails",
        JSON.stringify({
          value: dataFetcher.yamlFileDetails,
          expiresOn: Date.now() + 1000 * 60 * 60 * 24, // 1day in ms
        })
      );

      renderUI.renderPage(dataFetcher);
    }

    loadingFilters.style.display = loadingPagination.style.display = "none";
    renderUI.attachPaginatedEventListeners(dataFetcher);
    renderUI.attachSearchBarEventListeners(dataFetcher);
  })();
};

document.addEventListener("DOMContentLoaded", async () => {
  const dataFetcher = new DataFetcher();
  const renderUI = new RenderUI();

  const localStorageContent = JSON.parse(
    localStorage.getItem("OnnxYamlFileDetails")
  );

  if (!localStorageContent) {
    loadPage(renderUI, dataFetcher);
  } else {
    const expiresOn = JSON.parse(
      localStorage.getItem("OnnxYamlFileDetails")
    ).expiresOn;

    if (Date.now() > expiresOn) {
      localStorage.removeItem("OnnxYamlFileDetails");
      loadPage(renderUI, dataFetcher);
    } else {
      dataFetcher.yamlFileDetails = localStorageContent.value;
      dataFetcher.updateAvailableFilters(dataFetcher.yamlFileDetails);

      renderUI.renderPage(dataFetcher, true);
      renderUI.attachPaginatedEventListeners(dataFetcher);
      renderUI.attachSearchBarEventListeners(dataFetcher);
    }
  }
});
