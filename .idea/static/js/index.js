

// 获取左侧导航栏按钮元素和右侧功能区的各个部分
const sidebarLinks = document.querySelectorAll('.sidebar a');
const contentSections = document.querySelectorAll('.content > div');

// 监听右侧功能区的滚动事件
document.querySelector('.content').addEventListener('scroll', () => {
    const currentScroll = document.querySelector('.content').scrollTop;
    console.log('sccccc')
    contentSections.forEach((section, index) => {
        const sectionTop = section.offsetTop;
        const sectionBottom = sectionTop + section.clientHeight;

        if (currentScroll >= sectionTop && currentScroll < sectionBottom) {
            // 更新按钮样式
            sidebarLinks.forEach(link => link.classList.remove('active'));
            sidebarLinks[index].classList.add('active');
        }
    });
});

// function normSubmit() {
//     const form = document.getElementById("normForm");
//     // const resultTextBox = document.getElementById("normParaTemp");
//     const formData = new FormData(form);
//     console.log(formData)
//     fetch("/norm_form", {
//         method: "POST",
//         body: formData,
//     })
//     .then(response => response.text())
//     .then(data => {
//         // 将后端处理结果显示在文本框中
//         console.log(data)
//         // resultTextBox.value = data;
//     })
//     .catch(error => {
//         console.error("Error:", error);
//     });
// }
const normForm = document.getElementById("normForm")
normForm.onsubmit = async (e) => {
    // const datadirText = document.getElementById('datadirText')
    // normForm.append('datadir', datadirText.value)
    e.preventDefault();
    // console.log(e)
    let response = await fetch('/norm_form', {
      method: 'POST',
      body: new FormData(normForm)
    });
    let result = await response.json();
    alert(result);
};
// value="归一化（自动保存至$data_root_dir/vars文件夹）"

const selectInputTifBtn = document.getElementById('selectInputTifBtn')
  selectInputTifBtn.addEventListener('click', () => {
    // 打开文件对话框
    const { dialog } = require('electron').remote;
    dialog.showOpenDialog({ properties: ['openFile'] }).then(result => {
      if (!result.canceled && result.filePaths.length > 0) {
        const selectedFilePath = result.filePaths[0];
        // 将文件路径传递给后端或在前端进行进一步处理
        console.log('选择的文件路径：', selectedFilePath);
      }
    });
  });