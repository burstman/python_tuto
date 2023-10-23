import numpy as np

grades = [85, 90, 88, 92, 95, 80, 75, 98, 89, 83]
np_grades = np.array(grades)

print("Grades Mean: ", np.mean(np_grades))
print("Grades Median", np.median(np_grades))
print("Grades Standard Deviation:", np.std(np_grades))
print("Grades Max:", np.max(np_grades))
print("Grades Sorted in Ascending Order:", np.sort(np_grades)[::-1])
print("Index of the highest Grade", np.argmax(np_grades))
print("Number of students that are above 90:", len(np_grades[np_grades > 90]))
print("Percentage of student scored above 90:", np.mean([np_grades > 90]*100))
print("Percentage of student scored below 75:", np.mean([np_grades < 75]*100))
high_performers = np_grades[np_grades > 90]
passing_grades = np_grades[np_grades > 75]
print("high performers:", high_performers)
print("passing grades:",passing_grades)
