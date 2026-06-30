from tools.excel_reader_tool import excel_reader_tool

result = excel_reader_tool.invoke({
    "xlsx_path": "output/reliance.xlsx"
})

print(result)