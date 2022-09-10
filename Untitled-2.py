#!/usr/bin/env python3
import time

about_me = {
    'Name': 'Karjout Abdeslam',
    'Nickname': 'Bitquark',
    'Age': 24,
    'i love': 'cats🐈🐈 and Demons    😈 😈 ',
    'msg for u': 'Lifes goal is to finish, so do what you want before it ends. 😊❤️'
}


def main():
    for t, i in about_me.items():
        time.sleep(2)
        print(f'{t}: {i}')
    print('\nBye ;D')


if __name__ == '__main__':
    main()
