#!/bin/bash
# Скрипт для перезагрузки IntelliSense в VS Code

echo "🔄 Перезагрузка IntelliSense для VS Code..."
echo ""

# 1. Удалить кэш C++ extension
echo "1. Очистка кэша C++ extension..."
rm -rf ~/.cache/vscode-cpptools 2>/dev/null
rm -rf ~/.vscode/extensions/ms-vscode.cpptools-*/ipch 2>/dev/null
echo "   ✅ Кэш очищен"

# 2. Проверить compile_commands.json
echo ""
echo "2. Проверка compile_commands.json..."
if [ -L compile_commands.json ]; then
    echo "   ✅ Симлинк существует"
    if [ -f build/compile_commands.json ]; then
        echo "   ✅ Файл build/compile_commands.json существует"
    else
        echo "   ⚠️  Файл build/compile_commands.json не найден"
        echo "   💡 Запустите: cmake --preset linux-main"
    fi
else
    echo "   ⚠️  Симлинк не найден"
    if [ -f build/compile_commands.json ]; then
        echo "   💡 Создаю симлинк..."
        ln -sf build/compile_commands.json compile_commands.json
        echo "   ✅ Симлинк создан"
    fi
fi

# 3. Проверить c_cpp_properties.json
echo ""
echo "3. Проверка c_cpp_properties.json..."
if [ -f .vscode/c_cpp_properties.json ]; then
    echo "   ✅ Файл существует"
else
    echo "   ❌ Файл не найден!"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Готово!"
echo ""
echo "📋 Следующие шаги:"
echo "   1. Закройте VS Code полностью"
echo "   2. Откройте VS Code заново"
echo "   3. Откройте src/main.cpp"
echo "   4. Подождите несколько секунд (индексация)"
echo ""
echo "💡 Если не помогло:"
echo "   - Нажмите Ctrl+Shift+P"
echo "   - Введите: C/C++: Select a Configuration..."
echo "   - Выберите 'Linux'"
echo "═══════════════════════════════════════════════════════════"

