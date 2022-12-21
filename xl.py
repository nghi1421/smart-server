# import xlsxwriter module
import xlsxwriter

# Workbook() takes one, non-optional, argument
# which is the filename that we want to create.
workbook = xlsxwriter.Workbook('dts-phuclong.xlsx')

# The workbook object is then used to add new
# worksheet via the add_worksheet() method.
worksheet = workbook.add_worksheet()

row = 0
column = 1

c1 = ["tôi", "mình", "tui", "tớ", "mị", "lão gia", "bổn cô nương", "tại hạ", "ngộ", "trẫm", "em", "anh"
      ]
c2 = ["đang", "hiện đang", "bây giờ đang", "hiện tại đang"]
c3 = ["rất", "quá", "cực kì", "thật"]
c4 = ["vui", "yêu đời", "hứng khởi", "hân hoan", "tích cực", "khó chịu", "thất vọng", "chán nản", "bực", "ức chế", "trầm cảm", "buồn", "rầu", "suy sụp", "sụp đổ", "nản"," chán", "mông lung"
      ]
c5 = ["quá", "cực kì", "lắm", "thật"]
c6 = ["tương tư", "thầm nhớ", "nghĩ về"]
c7 = ["người yêu", "một người", "một người con gái", "một người con trai", "một bạn cùng lớp", "một anh hàng xóm", "một chị gái", "một em"]
# iterating through content list
for s in c1:
    # write operation perform
    for be in c2:
            for adv in c3:
                for i,adj in enumerate(c4):
                    content = s+' '+be+' '+adv+' '+adj
                    worksheet.write(row, 0, content)
                    if i < 4:
                        worksheet.write(row, column, "pos")
                    else:
                        worksheet.write(row, column, "nev")
                    row += 1
            for i, adj in enumerate(c4):
                for adv in c5:
                    content = s + ' ' + be + ' ' + adj + ' ' + adv
                    worksheet.write(row, 0, content)
                    if i < 4:
                        worksheet.write(row, column, "pos")
                    else:
                        worksheet.write(row, column, "nev")
                    row += 1
            for v in c6:
                for o in c7:
                    content = s + ' ' + be + ' ' + v + ' ' +o
                    worksheet.write(row, 0, content)
                    worksheet.write(row, column, "pos")
                    row += 1
workbook.close()