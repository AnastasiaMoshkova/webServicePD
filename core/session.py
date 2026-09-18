"""
Идентификатор сессии для изоляции состояния между одновременными пользователями.

Модальности (мимика, рука) раньше хранили состояние обработки (пути к
временным файлам, объект записи видео и т.п.) в одном общем сервисе на всё
приложение — при двух одновременных пользователях они конфликтовали друг
с другом (один затирал файлы/запись другого). Сессия даёт каждому браузеру
свой изолированный слой: свою подпапку с файлами и свою запись в словаре
состояний сервиса.
"""

import uuid
from typing import Optional

from fastapi import Request, Response, WebSocket

SESSION_COOKIE_NAME = "session_id"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24  # сутки


def new_session_id() -> str:
    return uuid.uuid4().hex


def read_session_id(request: Request) -> Optional[str]:
    return request.cookies.get(SESSION_COOKIE_NAME)


def read_or_new_session_id(request: Request) -> str:
    """Для начала обработчика: взять cookie, если её нет — сгенерировать новую."""
    return read_session_id(request) or new_session_id()


def set_session_cookie(response: Response, session_id: str) -> None:
    """В конце обработчика: проставить/обновить cookie на итоговом ответе."""
    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )


def get_or_create_session_id(request: Request, response: Response) -> str:
    """Удобный вызов в один шаг для простых GET-обработчиков."""
    session_id = read_or_new_session_id(request)
    set_session_cookie(response, session_id)
    return session_id


def get_session_id_from_websocket(websocket: WebSocket) -> str:
    """
    Для WebSocket: cookie можно только прочитать (не выставить на хендшейке).
    В обычном сценарии cookie уже есть — страница мимики/руки открывается
    через GET, который её ставит, а WebSocket-соединение открывается уже после.
    Если cookie почему-то нет — используем разовый id только на это соединение.
    """
    return websocket.cookies.get(SESSION_COOKIE_NAME) or new_session_id()
