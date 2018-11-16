def main():
    a = 3
    b = 4
    c = -10

    rate = .1

    count = 0
    for i in range(1, 100):
        a = a + rate
        b = b + rate
        c = c + rate

        temp = a + b + c

        if temp >= 0:
            break

        count += 1

    print(count)


if __name__ == '__main__':
    main()
