"""Role-based access control primitives used by every data-layer function.

UI pages never decide access on their own: each data function receives the
calling ``Actor`` and raises ``PermissionDenied`` if the role does not allow
the operation (Section 7 of the requirements).
"""
from __future__ import annotations

from dataclasses import dataclass

ROLE_STUDENT = "student"
ROLE_ADMIN = "admin"


class PermissionDenied(Exception):
    pass


@dataclass(frozen=True)
class Actor:
    id: int
    username: str
    role: str
    full_name: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN


def require_actor(actor: Actor | None) -> Actor:
    if actor is None or not isinstance(actor, Actor):
        raise PermissionDenied("Authentication required.")
    return actor


def require_admin(actor: Actor | None) -> Actor:
    actor = require_actor(actor)
    if not actor.is_admin:
        raise PermissionDenied("Administrator privileges required.")
    return actor


def require_self_or_admin(actor: Actor | None, user_id: int) -> Actor:
    actor = require_actor(actor)
    if not actor.is_admin and int(actor.id) != int(user_id):
        raise PermissionDenied("You can only access your own data.")
    return actor


def scope_user_id(actor: Actor | None, requested_user_id: int | None) -> int | None:
    """Resolve which user's data a query may return.

    Students are always pinned to themselves regardless of what was requested;
    admins may request any user or ``None`` (= all users).
    """
    actor = require_actor(actor)
    if actor.is_admin:
        return requested_user_id
    if requested_user_id is not None and int(requested_user_id) != int(actor.id):
        raise PermissionDenied("You can only access your own data.")
    return actor.id
