import matplotlib.pyplot as plt

# Đọc dữ liệu từ tập tin và chuyển nó thành các mảng
data = []
with open('tron.txt', 'r') as file:
    for line in file:
        x, y = map(int, line.split())
        data.append((x, y))

x_values = [point[0] for point in data]
y_values = [point[1] for point in data]

plt.plot(x_values, y_values)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Biểu đồ')
plt.show()
